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
    // Safety check
    if (b.size() != dim) return std::numeric_limits<float>::infinity();
    
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

    // Parse Filter
    if (input.named_parameters.find("allowed_ids") != input.named_parameters.end()) {
        result->has_filter = true;
        auto filter_value = input.named_parameters["allowed_ids"];
        auto &filter_children = ListValue::GetChildren(filter_value);
        for (auto &child : filter_children) {
            result->allowed_ids.insert(child.GetValue<int64_t>());
        }
    }

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
        throw std::runtime_error("Failed to load centroids: " + (result ? result->GetError() : "Unknown error"));
    }

    while (auto chunk = result->Fetch()) {
        if (!chunk || chunk->size() == 0) break;
        
        auto &vector_col = chunk->data[0];
        auto row_count = chunk->size();
        
        // FIX: Infer dimension from the data itself for the first batch, or hardcode 128
        // Here we look at the child vector length vs row count
        auto &child_vec = ListVector::GetEntry(vector_col);
        idx_t child_len = ListVector::GetListSize(vector_col);
        idx_t array_size = (row_count > 0) ? (child_len / row_count) : 128; // Fallback
        
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

    state->has_filter = bind_data.has_filter;
    state->allowed_ids = bind_data.allowed_ids;
    
    if (state->has_filter) {
        printf("Hybrid Filter Active: Restricted search to %zu specific IDs.\n", state->allowed_ids.size());
    }

    auto &db = context.db;
    Connection new_connection(*db);
    auto &new_context = *new_connection.context;

    try {
        LoadCentroids(new_context, bind_data.index_name, state->centroids);
        // printf("Loaded %zu centroids.\n", state->centroids.size());
    } catch (std::exception &e) {
        printf("Error loading index: %s\n", e.what());
    }

    return std::move(state);
}

void ComputeIVFSearch(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
    auto &state = data_p.global_state->Cast<IVFSearchGlobalState>();
    auto &bind_data = data_p.bind_data->Cast<IVFSearchFunctionData>();

    if (state.current_offset == 0 && state.final_results.empty() && !state.is_done) {
        
        // A. PROBE
        std::vector<std::pair<float, int>> cluster_distances;
        for (int i = 0; i < (int)state.centroids.size(); i++) {
            float dist = L2SquaredDistance(bind_data.query_vector, state.centroids[i]);
            cluster_distances.push_back({dist, i});
        }
        std::sort(cluster_distances.begin(), cluster_distances.end());

        std::vector<int> probes;
        int actual_nprobe = std::min(bind_data.nprobe, (int32_t)cluster_distances.size());
        for (int i = 0; i < actual_nprobe; i++) {
            probes.push_back(cluster_distances[i].second);
        }

        // B. SCAN
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

        auto &db = context.db;
        Connection search_conn(*db);
        auto query_result = search_conn.context->Query(sql, false);

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

                // --- FIX: Use query vector size as truth ---
                idx_t array_size = bind_data.query_vector.size(); 
                
                auto &child_vec = ListVector::GetEntry(vec_col);
                child_vec.Flatten(row_count * array_size);
                auto float_data = FlatVector::GetData<float>(child_vec);

                for (idx_t i = 0; i < row_count; i++) {
                    int64_t candidate_id = id_data[i];

                    if (state.has_filter) {
                        if (state.allowed_ids.find(candidate_id) == state.allowed_ids.end()) {
                            continue; 
                        }
                    }

                    std::vector<float> candidate_vec;
                    candidate_vec.reserve(array_size);
                    for (idx_t j = 0; j < array_size; j++) {
                        candidate_vec.push_back(float_data[i * array_size + j]);
                    }

                    float dist = L2SquaredDistance(bind_data.query_vector, candidate_vec);

                    if (top_k_heap.size() < (size_t)bind_data.k) {
                        top_k_heap.push({dist, candidate_id});
                    } else if (dist < top_k_heap.top().first) {
                        top_k_heap.pop();
                        top_k_heap.push({dist, candidate_id});
                    }
                }
            }
        }

        while (!top_k_heap.empty()) {
            auto item = top_k_heap.top();
            state.final_results.push_back({item.second, item.first});
            top_k_heap.pop();
        }
        std::reverse(state.final_results.begin(), state.final_results.end());
        state.is_done = true;
    }

    idx_t remaining = state.final_results.size() - state.current_offset;
    if (remaining == 0) {
        output.SetCardinality(0);
        return;
    }

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