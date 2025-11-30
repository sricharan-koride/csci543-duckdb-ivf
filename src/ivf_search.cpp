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
#include "pq.hpp"

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
    
    // Parse Query Vector (accept either a fixed-size ARRAY or a variable-length LIST)
    auto query_val = input.inputs[1];
    const vector<Value> *children_ptr = nullptr;
    if (query_val.type().id() == LogicalTypeId::LIST) {
        children_ptr = &ListValue::GetChildren(query_val);
    } else {
        children_ptr = &ArrayValue::GetChildren(query_val);
    }
    const auto &children = *children_ptr;
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
    // Optional WHERE clause pushed by the optimizer
    if (input.named_parameters.find("where_clause") != input.named_parameters.end()) {
        result->has_where = true;
        result->where_clause = input.named_parameters["where_clause"].GetValue<string>();
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
    state->has_where = bind_data.has_where;
    state->where_clause = bind_data.where_clause;
    
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

    // Load base table name from metadata table if present
    try {
        auto meta_q = StringUtil::Format("SELECT table_name FROM ivf_index_metadata WHERE index_name = '%s' LIMIT 1", bind_data.index_name.c_str());
        auto meta_res = new_context.Query(meta_q, false);
        if (meta_res && meta_res->GetError().empty()) {
            while (auto chunk = meta_res->Fetch()) {
                if (!chunk || chunk->size() == 0) break;
                auto &col = chunk->data[0];
                auto row_count = chunk->size();
                // Read first row's string
                auto str_data = FlatVector::GetData<string_t>(col);
                if (row_count > 0) {
                    state->base_table = str_data[0].GetString();
                    break;
                }
            }
        }
    } catch (...) {
        // ignore metadata load failures; defaulting to empty -> caller must handle
    }

    return std::move(state);
}

void ComputeIVFSearch(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
    auto &state = data_p.global_state->Cast<IVFSearchGlobalState>();
    auto &bind_data = data_p.bind_data->Cast<IVFSearchFunctionData>();

    // --- STEP 1: Perform Search (Only once) ---
    if (state.current_offset == 0 && state.final_results.empty() && !state.is_done) {
        
        // A. PROBE: Calculate distance to ALL centroids first
        std::vector<std::pair<float, int>> cluster_distances;
        for (int i = 0; i < (int)state.centroids.size(); i++) {
            float dist = L2SquaredDistance(bind_data.query_vector, state.centroids[i]);
            cluster_distances.push_back({dist, i});
        }
        // Sort centroids by distance (closest first)
        std::sort(cluster_distances.begin(), cluster_distances.end());

        // --- IVF-PQ: Load PQ structures and build LUT for this query (if available) ---
        PQCodebook pq_codebook;
        std::vector<std::vector<uint8_t>> pq_codes;
        std::vector<std::vector<float>> pq_lut;
        bool pq_available = false;

        try {
            // Load PQ codebooks and codes using the same context
            LoadPQCodebooks(context, bind_data.index_name, pq_codebook);
            LoadPQCodes(context, bind_data.index_name, pq_codes);
            BuildPQDistanceLUT(bind_data.query_vector, pq_codebook, pq_lut);
            pq_available = true;
            // printf("IVF-PQ enabled: M=%d, Ks=%d, codes=%zu\n", pq_codebook.M, pq_codebook.Ks, pq_codes.size());
        } catch (std::exception &e) {
            // If anything fails (no PQ tables, etc.), fall back to exact L2
            printf("IVF-PQ disabled (falling back to exact L2): %s\n", e.what());
        }

        // Priority Queue for Top-K results
        using ResultPair = std::pair<float, int64_t>;
        std::priority_queue<ResultPair> top_k_heap;

        // --- ADAPTIVE SCAN LOOP ---
        size_t clusters_scanned = 0;
        size_t total_clusters = cluster_distances.size();
        int current_nprobe_increment = bind_data.nprobe; // Start with user's nprobe

        while (clusters_scanned < total_clusters) {
            
            // 1. Identify which clusters to scan in this batch
            size_t end_idx = std::min(clusters_scanned + current_nprobe_increment, total_clusters);
            if (end_idx == clusters_scanned) break;

            std::vector<int> batch_clusters;
            string cluster_list_str;
            for(size_t i = clusters_scanned; i < end_idx; i++) {
                int cid = cluster_distances[i].second;
                batch_clusters.push_back(cid);
                cluster_list_str += std::to_string(cid);
                if (i < end_idx - 1) cluster_list_str += ",";
            }

            // Debug: See it adapting
            // printf("Adaptive Scan: Checking clusters rank %zu to %zu...\n", clusters_scanned, end_idx);

            // 2. Execute SQL for this batch
            // Choose base table from metadata if available, otherwise fall back to 'sift_base'
            string base_table = state.base_table.empty() ? string("sift_base") : state.base_table;
            string sql = StringUtil::Format(
                "SELECT s.id::BIGINT, s.vec FROM %s s JOIN "
                "(SELECT unnest(vector_ids) as vid FROM ivf_lists_%s WHERE cluster_id IN (%s)) c "
                "ON s.id = c.vid", 
                base_table.c_str(), bind_data.index_name.c_str(), cluster_list_str.c_str()
            );
            // Append optional where clause (safe-guarded by optimizer)
            if (state.has_where && !state.where_clause.empty()) {
                sql += " WHERE ";
                sql += state.where_clause;
            }
            auto &db = context.db;
            Connection search_conn(*db);
            auto query_result = search_conn.context->Query(sql, false);

            // 3. Process Candidates
            if (query_result && query_result->GetError().empty()) {
                while (auto chunk = query_result->Fetch()) {
                    if (!chunk || chunk->size() == 0) break;
                    chunk->Flatten();

                    auto &id_col = chunk->data[0];
                    auto &vec_col = chunk->data[1];
                    auto row_count = chunk->size();
                    auto id_data = FlatVector::GetData<int64_t>(id_col);

                    idx_t array_size = bind_data.query_vector.size(); 
                    auto &child_vec = ListVector::GetEntry(vec_col);
                    child_vec.Flatten(row_count * array_size);
                    auto float_data = FlatVector::GetData<float>(child_vec);

                    for (idx_t i = 0; i < row_count; i++) {
                        int64_t candidate_id = id_data[i];

                        // Gatekeeper (Filter)
                        if (state.has_filter) {
                            if (state.allowed_ids.find(candidate_id) == state.allowed_ids.end()) {
                                continue; 
                            }
                        }

                        // Distance Calculation: prefer IVF-PQ, fall back to exact L2 if needed
                        float dist = 0.0f;
                        bool used_pq = false;

                        if (pq_available &&
                            candidate_id >= 0 &&
                            static_cast<idx_t>(candidate_id) < pq_codes.size() &&
                            !pq_codes[static_cast<idx_t>(candidate_id)].empty()) {
                            // Use PQ distance via LUT
                            dist = PQDistance(pq_codes[static_cast<idx_t>(candidate_id)], pq_lut);
                            used_pq = true;
                        } else {
                            // Fallback: reconstruct full vector and compute exact L2 distance
                            std::vector<float> candidate_vec;
                            candidate_vec.reserve(array_size);
                            for (idx_t j = 0; j < array_size; j++) {
                                candidate_vec.push_back(float_data[i * array_size + j]);
                            }
                            dist = L2SquaredDistance(bind_data.query_vector, candidate_vec);
                        }

                        if (top_k_heap.size() < (size_t)bind_data.k) {
                            top_k_heap.push({dist, candidate_id});
                        } else if (dist < top_k_heap.top().first) {
                            top_k_heap.pop();
                            top_k_heap.push({dist, candidate_id});
                        }
                    }
                }
            }

            // 4. Adaptive Check: Do we have enough results?
            clusters_scanned = end_idx;
            if (top_k_heap.size() >= (size_t)bind_data.k) {
                // Success! We found k valid items. Stop scanning.
                // printf("Adaptive Search: Found %zu results. Stopping early.\n", top_k_heap.size());
                break;
            } else {
                // Not enough results. Expand search!
                // printf("Adaptive Search: Only found %zu results. Expanding search...\n", top_k_heap.size());
                // We keep the increment same (scan next 5, then next 5...)
            }
        }

        // Finalize Results
        while (!top_k_heap.empty()) {
            auto item = top_k_heap.top();
            state.final_results.push_back({item.second, item.first});
            top_k_heap.pop();
        }
        std::reverse(state.final_results.begin(), state.final_results.end());
        state.is_done = true;
    }

    // --- STEP 2: Stream Results ---
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