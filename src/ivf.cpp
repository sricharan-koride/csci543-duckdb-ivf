#include "ivf.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"
#include <vector>
#include "duckdb/main/database.hpp"
#include "duckdb/main/connection.hpp"
#include "pq.hpp"
#include <cstdint>

// ADD THESE HEADERS FOR K-MEANS 
#include "kmeans/Kmeans.hpp"
#include "kmeans/SimpleMatrix.hpp"
#include <random>

#include "duckdb/main/appender.hpp"
#include "duckdb/common/types/value.hpp"

#include <cmath>   
#include <limits>  
#include <map>


namespace duckdb {
namespace {

// Helper function to compute L2 (Euclidean) distance
// We use squared distance to avoid the expensive sqrt()
float L2SquaredDistance(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0;
    size_t dim = a.size();
    for (size_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// Pointer-based helper for PQ encoding
float L2SquaredDistancePtr(const float *a, const float *b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

}

void CreateIVFIndex(DataChunk &args, ExpressionState &state, Vector &result) {
    // Get the arguments
    auto index_name = args.GetValue(0, 0).ToString();
    auto table_name = args.GetValue(1, 0).ToString();
    auto column_name = args.GetValue(2, 0).ToString();
    printf("CreateIVFIndex called:\n");
    printf("  Index Name: %s\n", index_name.c_str());
    printf("  Table Name: %s\n", table_name.c_str()); 
    printf("  Column Name: %s\n", column_name.c_str());

    // Create a new, independent connection
    auto &db = state.GetContext().db;
    Connection new_connection(*db);
    auto &context = *new_connection.context;

    // Construct and run the sample query
    auto sample_query = StringUtil::Format(
        "SELECT %s FROM %s USING SAMPLE 10 PERCENT (BERNOULLI)",
        column_name, table_name
    );
    printf("Running query: %s\n", sample_query.c_str());

    auto query_result = context.Query(sample_query, false);

    // Check for query error
    if (!query_result || !query_result->GetError().empty()) {
        printf("Error: %s\n", query_result ? query_result->GetError().c_str() : "null result");
        result.SetVectorType(VectorType::CONSTANT_VECTOR);
        ConstantVector::SetNull(result, true);
        return;
    }

    // Vector to hold sampled data
    std::vector<std::vector<float>> sample_vectors;
    printf("Successfully fetched sample vectors. Processing...\n");

    // Loop over results and extract vectors
    while (auto chunk = query_result->Fetch()) {
        if (!chunk || chunk->size() == 0) {
            break;
        }

        auto &vector_column = chunk->data[0];
        auto logical_type = vector_column.GetType().id();
        auto row_count = chunk->size();

        if (logical_type == LogicalTypeId::LIST) {
            // Original LIST handling
            auto list_data = ListVector::GetData(vector_column);
            auto &child_vector = ListVector::GetEntry(vector_column);
            auto float_data = FlatVector::GetData<float>(child_vector);

            for (idx_t i = 0; i < row_count; i++) {
                auto list_entry = list_data[i];
                auto offset = list_entry.offset;
                auto length = list_entry.length;
                std::vector<float> row_vector;
                row_vector.reserve(length);
                for (idx_t j = 0; j < length; j++) {
                    row_vector.push_back(float_data[offset + j]);
                }
                sample_vectors.push_back(std::move(row_vector));
            }

        } else if (logical_type == LogicalTypeId::ARRAY) {
            auto array_size = ArrayType::GetSize(vector_column.GetType());
            
            // ARRAYs also use ListVector::GetEntry to get the child
            auto &child_vector = ListVector::GetEntry(vector_column);
            
            // Flatten the child vector to ensure we can read it
            auto total_child_elements = row_count * array_size;
            child_vector.Flatten(total_child_elements);

            auto float_data = FlatVector::GetData<float>(child_vector);

            for (idx_t i = 0; i < row_count; i++) {
                auto offset = i * array_size;
                std::vector<float> row_vector;
                row_vector.reserve(array_size);
                for (idx_t j = 0; j < array_size; j++) {
                    row_vector.push_back(float_data[offset + j]);
                }
                sample_vectors.push_back(std::move(row_vector));
            }

        } else {
            printf("Error: Unsupported vector type %s\n",
                   LogicalTypeIdToString(logical_type).c_str());
        }
    }

    printf("Vector sampling complete. Total vectors sampled: %zu\n", sample_vectors.size());

    // Perform K-Means Clustering
    
    // Flatten our vector data for the library
    size_t total_vectors = sample_vectors.size();
    size_t dim = 0;
    // fallback to a full-table scan to collect vectors for k-means.
    if (total_vectors == 0) {
        printf("Warning: Sampling returned 0 vectors; performing full-table scan to collect vectors for clustering...\n");
        auto full_sample_query_str = StringUtil::Format("SELECT %s FROM %s", column_name, table_name);
        auto full_sample_query = context.Query(full_sample_query_str, false);
        if (!full_sample_query || !full_sample_query->GetError().empty()) {
            printf("Error: Failed to collect vectors from full table: %s\n", full_sample_query ? full_sample_query->GetError().c_str() : "null result");
            result.SetVectorType(VectorType::CONSTANT_VECTOR);
            ConstantVector::SetNull(result, true);
            return;
        }
        while (auto chunk = full_sample_query->Fetch()) {
            if (!chunk || chunk->size() == 0) break;
            auto &vector_column = chunk->data[0];
            auto logical_type = vector_column.GetType().id();
            auto row_count = chunk->size();

            if (logical_type == LogicalTypeId::LIST) {
                auto list_data = ListVector::GetData(vector_column);
                auto &child_vector = ListVector::GetEntry(vector_column);
                auto float_data = FlatVector::GetData<float>(child_vector);
                for (idx_t i = 0; i < row_count; i++) {
                    auto list_entry = list_data[i];
                    auto offset = list_entry.offset;
                    auto length = list_entry.length;
                    std::vector<float> row_vector;
                    row_vector.reserve(length);
                    for (idx_t j = 0; j < length; j++) {
                        row_vector.push_back(float_data[offset + j]);
                    }
                    sample_vectors.push_back(std::move(row_vector));
                }

            } else if (logical_type == LogicalTypeId::ARRAY) {
                auto array_size = ArrayType::GetSize(vector_column.GetType());
                auto &child_vector = ListVector::GetEntry(vector_column);
                auto total_child_elements = row_count * array_size;
                child_vector.Flatten(total_child_elements);
                auto float_data = FlatVector::GetData<float>(child_vector);
                for (idx_t i = 0; i < row_count; i++) {
                    auto offset = i * array_size;
                    std::vector<float> row_vector;
                    row_vector.reserve(array_size);
                    for (idx_t j = 0; j < array_size; j++) {
                        row_vector.push_back(float_data[offset + j]);
                    }
                    sample_vectors.push_back(std::move(row_vector));
                }
            }
        }
        total_vectors = sample_vectors.size();
        printf("Full-scan sampling complete. Total vectors sampled: %zu\n", total_vectors);
        if (total_vectors == 0) {
            printf("Error: No vectors available in table '%s'.\n", table_name.c_str());
            result.SetVectorType(VectorType::CONSTANT_VECTOR);
            ConstantVector::SetNull(result, true);
            return;
        }
    }
    dim = sample_vectors[0].size();

    std::vector<float> flat_data;
    flat_data.reserve(total_vectors * dim);
    for (const auto& vec : sample_vectors) {
        flat_data.insert(flat_data.end(), vec.begin(), vec.end());
    }

    // Define k-means parameters
    int num_clusters = args.GetValue(3, 0).GetValue<int32_t>();
    int num_threads = 4;   

    printf("Starting k-means clustering with %zu vectors...\n", total_vectors);
    printf("  Clusters: %d, Dimensions: %zu\n", num_clusters, dim);

    // Wrap our flat data in the library's SimpleMatrix class
    kmeans::SimpleMatrix<float, int> matrix(
        dim,           
        total_vectors, 
        flat_data.data() 
    );

    // Prepare the k-means++ initializer
    kmeans::InitializeKmeanspp<float, int, int> initializer;

    // Prepare the Lloyd refinement algorithm
    kmeans::RefineLloydOptions<float, int> refiner_options;
    refiner_options.max_iterations = 20; // Max 20 iterations
    kmeans::RefineLloyd<float, int, int> refiner(refiner_options); 

    // Run the k-means computation
    auto result_data = kmeans::compute(
        matrix,
        initializer,
        num_clusters,
        refiner, 
        num_threads
    );

    printf("K-means clustering complete.\n");
    printf("  Total Iterations: %d\n", result_data.iterations);
    
    if (!result_data.centers.empty()) {
        printf("  First centroid's first value: %f\n", result_data.centers[0]);
    }

    // PQ TRAINING & CODEBOOK PERSISTENCE
    // Configure PQ parameters
    PQMetadata pq_meta;
    pq_meta.dim = static_cast<int>(dim);
    pq_meta.M = 16;    // number of sub-vectors
    pq_meta.Ks = 256;  // number of centroids per subspace

    PQCodebook pq_codebook;
    try {
        printf("Starting PQ training (M = %d, Ks = %d, dim = %d)...\n", pq_meta.M, pq_meta.Ks, pq_meta.dim);
        TrainPQCodebooks(sample_vectors, pq_meta, pq_codebook);
        printf("PQ training complete. Persisting PQ codebooks...\n");
        PersistPQCodebooks(context, index_name, pq_codebook);
        printf("PQ codebooks persisted successfully.\n");
    } catch (std::exception &e) {
        printf("Error during PQ training/persistence: %s\n", e.what());
        result.SetVectorType(VectorType::CONSTANT_VECTOR);
        ConstantVector::SetNull(result, true);
        return;
    }

    // Persist Centroids
  
    
    auto centroid_table_name = "ivf_centroids_" + index_name;
    
    printf("Persisting %d centroids to table '%s'...\n", num_clusters, centroid_table_name.c_str());

    // Create the table
    auto create_table_sql = StringUtil::Format(
        "CREATE OR REPLACE TABLE %s (cluster_id INTEGER, centroid FLOAT[%llu])",
        centroid_table_name, dim
    );
    auto create_result = context.Query(create_table_sql, false);
    if (!create_result || !create_result->GetError().empty()) {
        printf("Error creating centroid table: %s\n", 
            create_result ? create_result->GetError().c_str() : "null result");
        result.SetVectorType(VectorType::CONSTANT_VECTOR);
        ConstantVector::SetNull(result, true);
        return;
    }

    // Use an Appender for high-speed inserts
    Appender appender(new_connection, "main", centroid_table_name);
    
    auto& centers_data = result_data.centers;

    for (int c = 0; c < num_clusters; c++) {
        // Get the start of the data for this centroid
        auto offset = c * dim;

        // Create a std::vector<Value> for the floats of this single centroid
        std::vector<Value> centroid_values;
        centroid_values.reserve(dim);
        for (size_t i = 0; i < dim; i++) {
            centroid_values.push_back(Value::FLOAT(centers_data[offset + i]));
        }

        // Create the DuckDB ARRAY value
        auto centroid_array = Value::ARRAY(LogicalType::FLOAT, std::move(centroid_values));

        // Append the row
        appender.BeginRow();
        appender.Append(Value::INTEGER(c));
        appender.Append(std::move(centroid_array));
        appender.EndRow();
    }
    
    // 4. Close the appender to flush changes
    appender.Close();
    
    printf("Centroids persisted successfully.\n");
   
    // This will hold our 100 centroids
    std::vector<std::vector<float>> centroids;
    // Create a query to read from the new table
    auto centroid_query = context.Query("SELECT centroid FROM " + centroid_table_name + " ORDER BY cluster_id", false);
    if (!centroid_query || !centroid_query->GetError().empty()) {
        printf("Error reading centroids: %s\n", centroid_query ? centroid_query->GetError().c_str() : "null result");
        result.SetVectorType(VectorType::CONSTANT_VECTOR);
        ConstantVector::SetNull(result, true);
        return;
    }

    // Loop over the results and extract the centroid vectors
    while (auto chunk = centroid_query->Fetch()) {
        if (!chunk || chunk->size() == 0) {
            break;
        }

        auto &vector_column = chunk->data[0];
        auto logical_type = vector_column.GetType().id();
        auto row_count = chunk->size();

        // This is the same ARRAY logic we used for sampling
        if (logical_type == LogicalTypeId::ARRAY) {
            auto array_size = ArrayType::GetSize(vector_column.GetType());
            auto &child_vector = ListVector::GetEntry(vector_column);
            auto total_child_elements = row_count * array_size;
            child_vector.Flatten(total_child_elements);
            auto float_data = FlatVector::GetData<float>(child_vector);

            for (idx_t i = 0; i < row_count; i++) {
                auto offset = i * array_size;
                std::vector<float> row_vector;
                row_vector.reserve(array_size);
                for (idx_t j = 0; j < array_size; j++) {
                    row_vector.push_back(float_data[offset + j]);
                }
                // Add the centroid to our in-memory list
                centroids.push_back(std::move(row_vector));
            }
        } else {
             printf("Error: Centroid table has unexpected type %s\n",
                   LogicalTypeIdToString(logical_type).c_str());
        }
    }
    
    printf("Successfully loaded %zu centroids back into memory.\n", centroids.size());

    // Create and Build Inverted Lists

    // Create the inverted list table
    auto inverted_list_table_name = "ivf_lists_" + index_name;
    printf("Creating inverted list table '%s'...\n", inverted_list_table_name.c_str());
    
    auto create_list_sql = StringUtil::Format(
        "CREATE OR REPLACE TABLE %s (cluster_id INTEGER, vector_ids BIGINT[])",
        inverted_list_table_name
    );
    auto create_list_result = context.Query(create_list_sql, false);
    if (!create_list_result || !create_list_result->GetError().empty()) {
        printf("Error creating inverted list table: %s\n", 
            create_list_result ? create_list_result->GetError().c_str() : "null result");
        result.SetVectorType(VectorType::CONSTANT_VECTOR);
        ConstantVector::SetNull(result, true);
        return;
    }

    // This map will hold our inverted lists in memory
    std::map<int, std::vector<int64_t>> inverted_lists;
    // Map
    std::map<int64_t, std::vector<uint8_t>> pq_codes;

    // Scan the *entire* base table
    printf("Scanning full table '%s' to build inverted lists...\n", table_name.c_str());
    // Force ID to be BIGINT so FlatVector::GetData<int64_t> works correctly
    auto full_scan_query_str = StringUtil::Format("SELECT id::BIGINT, %s FROM %s", column_name, table_name);
    auto full_scan_query = context.Query(full_scan_query_str, false);
    
    if (!full_scan_query || !full_scan_query->GetError().empty()) {
        printf("Error scanning full table: %s\n", full_scan_query ? full_scan_query->GetError().c_str() : "null result");
        result.SetVectorType(VectorType::CONSTANT_VECTOR);
        ConstantVector::SetNull(result, true);
        return;
    }
    
    int64_t total_vectors_processed = 0;

    // Loop over all data, chunk by chunk
    while (auto chunk = full_scan_query->Fetch()) {
        if (!chunk || chunk->size() == 0) {
            break;
        }

        auto &id_column = chunk->data[0];
        auto &vector_column = chunk->data[1];
        auto row_count = chunk->size();
        
        auto id_data = FlatVector::GetData<int64_t>(id_column);

        // Extract vectors (using the same ARRAY logic)
        auto array_size = ArrayType::GetSize(vector_column.GetType());
        auto &child_vector = ListVector::GetEntry(vector_column);
        auto total_child_elements = row_count * array_size;
        child_vector.Flatten(total_child_elements);
        auto float_data = FlatVector::GetData<float>(child_vector);

        std::vector<float> current_vector;
        current_vector.reserve(array_size);

        // Assign each vector to its nearest centroid and compute PQ codes
        for (idx_t i = 0; i < row_count; i++) {
            // Reconstruct the vector
            current_vector.clear();
            auto offset = i * array_size;
            for (idx_t j = 0; j < array_size; j++) {
                current_vector.push_back(float_data[offset + j]);
            }

            // Find the nearest centroid
            int best_cluster_id = -1;
            float min_dist = std::numeric_limits<float>::max();

            for (size_t c = 0; c < centroids.size(); c++) {
                float dist = L2SquaredDistance(current_vector, centroids[c]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster_id = static_cast<int>(c);
                }
            }

            // Add this vector's base-table ID to the correct inverted list
            auto vec_id = id_data[i];
            inverted_lists[best_cluster_id].push_back(vec_id);
            total_vectors_processed++;

            // Compute PQ code for this vector using the trained codebooks
            std::vector<uint8_t> pq_code;
            pq_code.reserve(pq_codebook.M);

            int subdim = pq_codebook.subvector_dim;
            for (int m = 0; m < pq_codebook.M; m++) {
                const float *cb = pq_codebook.codebooks[m].data();
                const float *slice = current_vector.data() + m * subdim;

                float best = std::numeric_limits<float>::max();
                uint8_t best_k = 0;

                for (int k = 0; k < pq_codebook.Ks; k++) {
                    float dist = L2SquaredDistancePtr(slice, cb + k * subdim, subdim);
                    if (dist < best) {
                        best = dist;
                        best_k = static_cast<uint8_t>(k);
                    }
                }

                pq_code.push_back(best_k);
            }

            // Store PQ code keyed by the base-table id
            pq_codes[vec_id] = std::move(pq_code);
        }
    }

        // Persist the inverted lists
        printf("Persisting inverted lists...\n");
        Appender list_appender(new_connection, "main", inverted_list_table_name);
        
        for (auto const& pair : inverted_lists) {
            auto cluster_id = pair.first;
            auto& id_list = pair.second;

            // Create a std::vector<Value> for the BIGINT IDs
            std::vector<Value> id_values;
            id_values.reserve(id_list.size());
            for (int64_t id : id_list) {
                id_values.push_back(Value::BIGINT(id));
            }

            // Create the DuckDB ARRAY value
            auto id_array = Value::ARRAY(LogicalType::BIGINT, std::move(id_values));

            // Append the row
            list_appender.BeginRow();
            list_appender.Append(Value::INTEGER(cluster_id));
            list_appender.Append(std::move(id_array));
            list_appender.EndRow();
        }
        
        list_appender.Close();
        printf("Inverted lists persisted successfully.\n");

    // Persist PQ codes (id BIGINT, code UBYTE[])
    printf("Persisting PQ codes...\n");
    auto pq_codes_table_name = "ivf_pq_codes_" + index_name;
    auto create_pqcodes_sql = StringUtil::Format(
        "CREATE TABLE IF NOT EXISTS %s (id BIGINT, code UTINYINT[])",
        pq_codes_table_name
    );
    auto create_pqcodes_result = context.Query(create_pqcodes_sql, false);
    if (!create_pqcodes_result || !create_pqcodes_result->GetError().empty()) {
        printf("Error creating PQ codes table: %s\n",
               create_pqcodes_result ? create_pqcodes_result->GetError().c_str() : "null result");
        result.SetVectorType(VectorType::CONSTANT_VECTOR);
        ConstantVector::SetNull(result, true);
        return;
    }

    {
        Appender pq_appender(new_connection, "main", pq_codes_table_name);
        for (auto &entry : pq_codes) {
            int64_t vec_id = entry.first;
            auto &code = entry.second;

            std::vector<Value> code_vals;
            code_vals.reserve(code.size());
            for (auto b : code) {
                // store as UTINYINT/UBYTE-friendly integer
                code_vals.push_back(Value::UTINYINT(b));
            }
            auto code_array = Value::ARRAY(LogicalType::UTINYINT, std::move(code_vals));

            pq_appender.BeginRow();
            pq_appender.Append(Value::BIGINT(vec_id));
            pq_appender.Append(std::move(code_array));
            pq_appender.EndRow();
        }
        pq_appender.Close();
    }

    printf("PQ codes persisted successfully.\n");

    // Persist index metadata for ann_search to discover the base table
    auto create_meta_sql = StringUtil::Format(
        "CREATE TABLE IF NOT EXISTS ivf_index_metadata (index_name VARCHAR, table_name VARCHAR, column_name VARCHAR)"
    );
    context.Query(create_meta_sql, false);
    auto insert_meta_sql = StringUtil::Format(
        "INSERT INTO ivf_index_metadata VALUES ('%s', '%s', '%s')",
        index_name.c_str(), table_name.c_str(), column_name.c_str()
    );
    context.Query(insert_meta_sql, false);

    printf("Finished scanning. Total vectors processed: %ld\n", total_vectors_processed);

    // Return a success message
    result.SetVectorType(VectorType::CONSTANT_VECTOR);
    auto data = ConstantVector::GetData<string_t>(result);
    data[0] = string_t("Index created successfully");
}

}