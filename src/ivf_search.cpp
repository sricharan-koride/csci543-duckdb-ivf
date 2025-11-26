#include "ivf_search.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/value_operations/value_operations.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/connection.hpp"
#include <algorithm>
#include <queue>
#include <cmath>

namespace duckdb {

// --- Helper: L2 Distance ---
static float L2SquaredDistance(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0;
    size_t dim = a.size();
    for (size_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

unique_ptr<FunctionData> BindIVFSearch(ClientContext &context, TableFunctionBindInput &input,
                                       vector<LogicalType> &return_types, vector<string> &names) {
    auto result = make_uniq<IVFSearchFunctionData>();
    result->index_name = input.inputs[0].ToString();
    
    // Parse Query Vector
    auto query_val = input.inputs[1];
    auto &children = ArrayValue::GetChildren(query_val);
    result->query_vector.reserve(children.size());
    for (auto &child : children) {
        result->query_vector.push_back(child.GetValue<float>());
    }

    result->k = input.inputs[2].GetValue<int32_t>();
    result->nprobe = input.inputs[3].GetValue<int32_t>();

    names.emplace_back("id");
    return_types.emplace_back(LogicalType::BIGINT);
    names.emplace_back("score");
    return_types.emplace_back(LogicalType::FLOAT);

    return std::move(result);
}

void LoadCentroids(ClientContext &context, const string &index_name, std::vector<std::vector<float>> &centroids) {
    string table_name = "ivf_centroids_" + index_name;
    auto query = "SELECT centroid FROM " + table_name + " ORDER BY cluster_id";
    auto result = context.Query(query, false);
    
    if (!result || !result->GetError().empty()) {
        // In a real app, handle error gracefully. For dev, we throw.
        throw std::runtime_error("Failed to load centroids: " + (result ? result->GetError() : "Unknown error"));
    }

    while (auto chunk = result->Fetch()) {
        if (!chunk || chunk->size() == 0) break;
        
        auto &vector_col = chunk->data[0];
        auto row_count = chunk->size();
        
        // Handle ARRAY type
        auto array_size = ArrayType::GetSize(vector_col.GetType());
        auto &child_vec = ListVector::GetEntry(vector_col);
        child_vec.Flatten(row_count * array_size);
        auto float_data = FlatVector::GetData<float>(child_vec);

        for (idx_t i = 0; i < row_count; i++) {
            std::vector<float> vec;
            vec.reserve(array_size);
            for (idx_t j = 0; j < array_size; j++) {
                vec.push_back(float_data[i * array_size + j]);
            }
            centroids.push_back(std::move(vec));
        }
    }
}

unique_ptr<GlobalTableFunctionState> InitIVFSearch(ClientContext &context, TableFunctionInitInput &input) {
    auto state = make_uniq<IVFSearchGlobalState>();
    auto &bind_data = input.bind_data->Cast<IVFSearchFunctionData>();

    auto &db = context.db;
    Connection new_connection(*db);
    auto &new_context = *new_connection.context;

    try {
        LoadCentroids(new_context, bind_data.index_name, state->centroids);
    } catch (std::exception &e) {
        printf("Error loading index: %s\n", e.what());
    }

    return std::move(state);
}

void ComputeIVFSearch(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
    auto &state = data_p.global_state->Cast<IVFSearchGlobalState>();
    auto &bind_data = data_p.bind_data->Cast<IVFSearchFunctionData>();

    // --- STEP 1: Perform Search (Only once) ---
    if (state.current_offset == 0 && state.final_results.empty()) {
        
        // A. PROBE: Find nearest centroids
        std::vector<std::pair<float, int>> cluster_distances;
        for (int i = 0; i < (int)state.centroids.size(); i++) {
            float dist = L2SquaredDistance(bind_data.query_vector, state.centroids[i]);
            cluster_distances.push_back({dist, i});
        }
        std::sort(cluster_distances.begin(), cluster_distances.end());

        // Get top nprobe clusters
        std::vector<int> probes;
        int actual_nprobe = std::min(bind_data.nprobe, (int32_t)cluster_distances.size());
        for (int i = 0; i < actual_nprobe; i++) {
            probes.push_back(cluster_distances[i].second);
        }

        // B. SCAN: Fetch candidates from DB using JOIN
        // Query: SELECT s.id, s.vec FROM sift_base s JOIN 
        //        (SELECT unnest(vector_ids) as vid FROM ivf_lists_... WHERE cluster_id IN (...)) c ON s.id=c.vid
        
        string cluster_list_str;
        for (size_t i = 0; i < probes.size(); i++) {
            cluster_list_str += std::to_string(probes[i]);
            if (i < probes.size() - 1) cluster_list_str += ",";
        }

        string sql = StringUtil::Format(
            "SELECT s.id::BIGINT, s.vec FROM sift_base s JOIN "
            "(SELECT unnest(vector_ids) as vid FROM ivf_lists_%s WHERE cluster_id IN (%s)) c "
            "ON s.id = c.vid", 
            bind_data.index_name.c_str(), cluster_list_str.c_str()
        );

        // We need a fresh connection for this query
        auto &db = context.db;
        Connection search_conn(*db);
        auto query_result = search_conn.context->Query(sql, false);

        // C. RANK: Compute exact distances for candidates
        // Priority queue to keep top K (Max-heap stores largest distance at top, so we can pop it)
        using ResultPair = std::pair<float, int64_t>;
        std::priority_queue<ResultPair> top_k_heap;

        if (query_result && query_result->GetError().empty()) {
            while (auto chunk = query_result->Fetch()) {
                if (!chunk || chunk->size() == 0) break;
                chunk->Flatten();
                auto &id_col = chunk->data[0];
                auto &vec_col = chunk->data[1];
                auto row_count = chunk->size();
                auto id_data = FlatVector::GetData<int64_t>(id_col);

                // Extract vectors
                auto array_size = ArrayType::GetSize(vec_col.GetType());
                auto &child_vec = ListVector::GetEntry(vec_col);
                child_vec.Flatten(row_count * array_size);
                auto float_data = FlatVector::GetData<float>(child_vec);

                for (idx_t i = 0; i < row_count; i++) {
                    // Reconstruct candidate vector
                    std::vector<float> candidate_vec;
                    candidate_vec.reserve(array_size);
                    for (idx_t j = 0; j < array_size; j++) {
                        candidate_vec.push_back(float_data[i * array_size + j]);
                    }

                    // Exact Distance
                    float dist = L2SquaredDistance(bind_data.query_vector, candidate_vec);

                    // Maintain Top-K
                    if (top_k_heap.size() < (size_t)bind_data.k) {
                        top_k_heap.push({dist, id_data[i]});
                    } else if (dist < top_k_heap.top().first) {
                        top_k_heap.pop();
                        top_k_heap.push({dist, id_data[i]});
                    }
                }
            }
        }

        // D. Finalize Results (Drain heap to vector)
        while (!top_k_heap.empty()) {
            auto item = top_k_heap.top();
            state.final_results.push_back({item.second, item.first});
            top_k_heap.pop();
        }
        // Heap gives worst-first, so reverse to get best-first
        std::reverse(state.final_results.begin(), state.final_results.end());
    }

    // --- STEP 2: Stream Results to Output ---
    idx_t remaining = state.final_results.size() - state.current_offset;
    if (remaining == 0) {
        output.SetCardinality(0); // Done!
        return;
    }

    // Send up to STANDARD_VECTOR_SIZE rows (usually 2048)
    idx_t count = std::min<idx_t>(remaining, STANDARD_VECTOR_SIZE);
    output.SetCardinality(count);

    auto &id_vec = output.data[0];
    auto &score_vec = output.data[1];
    auto id_ptr = FlatVector::GetData<int64_t>(id_vec);
    auto score_ptr = FlatVector::GetData<float>(score_vec);

    for (idx_t i = 0; i < count; i++) {
        auto &res = state.final_results[state.current_offset + i];
        id_ptr[i] = res.id;
        score_ptr[i] = res.score;
    }

    state.current_offset += count;
}

} // namespace duckdb