#include "ivf.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"
#include <vector>
#include "duckdb/main/database.hpp"
#include "duckdb/main/connection.hpp"

// --- ADD THESE HEADERS FOR K-MEANS ---
#include "kmeans/Kmeans.hpp"
#include "kmeans/SimpleMatrix.hpp"
#include <random>

#include "duckdb/main/appender.hpp"
#include "duckdb/common/types/value.hpp"

namespace duckdb {

void CreateIVFIndex(DataChunk &args, ExpressionState &state, Vector &result) {
    // 1. Get the arguments
    auto index_name = args.GetValue(0, 0).ToString();
    auto table_name = args.GetValue(1, 0).ToString();
    auto column_name = args.GetValue(2, 0).ToString();
    printf("CreateIVFIndex called:\n");
    printf("  Index Name: %s\n", index_name.c_str());
    printf("  Table Name: %s\n", table_name.c_str()); // <-- FIX 1: c_st() -> c_str()
    printf("  Column Name: %s\n", column_name.c_str());

    // 2. Create a new, independent connection
    auto &db = state.GetContext().db;
    Connection new_connection(*db);
    auto &context = *new_connection.context;

    // 3. Construct and run the sample query
    auto sample_query = StringUtil::Format(
        "SELECT %s FROM %s USING SAMPLE 10 PERCENT (BERNOULLI)",
        column_name, table_name
    );
    printf("Running query: %s\n", sample_query.c_str());

    auto query_result = context.Query(sample_query, false);

    // 4. Check for query error
    if (!query_result || !query_result->GetError().empty()) {
        printf("Error: %s\n", query_result ? query_result->GetError().c_str() : "null result");
        result.SetVectorType(VectorType::CONSTANT_VECTOR);
        ConstantVector::SetNull(result, true);
        return;
    }

    // 5. Vector to hold sampled data
    std::vector<std::vector<float>> sample_vectors;
    printf("Successfully fetched sample vectors. Processing...\n");

    // 6. Loop over results and extract vectors
    while (auto chunk = query_result->Fetch()) {
        if (!chunk || chunk->size() == 0) {
            break;
        }

        auto &vector_column = chunk->data[0];
        auto logical_type = vector_column.GetType().id();
        auto row_count = chunk->size();

        if (logical_type == LogicalTypeId::LIST) {
            // --- Original LIST handling ---
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
            // --- FIX 2: Correct ARRAY handling ---
            auto array_size = ArrayType::GetSize(vector_column.GetType());
            
            // ARRAYs also use ListVector::GetEntry to get the child
            auto &child_vector = ListVector::GetEntry(vector_column);
            
            // Flatten the child vector to ensure we can read it
            auto total_child_elements = row_count * array_size;
            child_vector.Flatten(total_child_elements);

            auto float_data = FlatVector::GetData<float>(child_vector);

            for (idx_t i = 0; i < row_count; i++) {
                auto offset = i * array_size; // Calculate offset
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

    // --- 7. ADDED: Perform K-Means Clustering ---
    
    // Flatten our vector data for the library
    size_t total_vectors = sample_vectors.size();
    size_t dim = 0;
    if (total_vectors == 0) {
        printf("Error: No vectors sampled.\n");
        result.SetVectorType(VectorType::CONSTANT_VECTOR);
        ConstantVector::SetNull(result, true);
        return;
    }
    dim = sample_vectors[0].size(); // Get dimensionality (e.g., 128)

    std::vector<float> flat_data;
    flat_data.reserve(total_vectors * dim);
    for (const auto& vec : sample_vectors) {
        flat_data.insert(flat_data.end(), vec.begin(), vec.end());
    }

    // Define k-means parameters
    int num_clusters = 100; // You can make this configurable later
    int num_threads = 4;    // Number of threads to use

    printf("Starting k-means clustering with %zu vectors...\n", total_vectors);
    printf("  Clusters: %d, Dimensions: %zu\n", num_clusters, dim);

    // 1. Wrap our flat data in the library's SimpleMatrix class
    kmeans::SimpleMatrix<float, int> matrix(
        dim,           // Number of dimensions (e.g., 128)
        total_vectors, // Number of vectors (e.g., 100642)
        flat_data.data() // Pointer to the raw float data
    );

    // 2. Prepare the k-means++ initializer
    // --- FIX 1: Provide all 3 template args: T, IDX, CLUSTER ---
    kmeans::InitializeKmeanspp<float, int, int> initializer;

    // 3. Prepare the Lloyd refinement algorithm
    // --- FIX 2: Create the RefineLloyd object ---
    kmeans::RefineLloydOptions<float, int> refiner_options;
    refiner_options.max_iterations = 20; // Max 20 iterations
    kmeans::RefineLloyd<float, int, int> refiner(refiner_options); // <-- Create the refiner

    // 4. Run the k-means computation
    auto result_data = kmeans::compute(
        matrix,
        initializer,
        num_clusters,
        refiner, // <-- Pass the refiner object, not the options
        num_threads
    );

    printf("K-means clustering complete.\n");
    printf("  Total Iterations: %d\n", result_data.iterations);
    
    // 'result_data.centers' is a std::vector<float> containing the centroids
    if (!result_data.centers.empty()) {
        printf("  First centroid's first value: %f\n", result_data.centers[0]);
    }

    // --- 8. Next Step: Persist Centroids ---
    // You would now write 'result_data.centers' to a new table.
  
    
    // We will create a table named 'ivf_centroids_[index_name]'
    auto centroid_table_name = "ivf_centroids_" + index_name;
    
    printf("Persisting %d centroids to table '%s'...\n", num_clusters, centroid_table_name.c_str());

    // 1. Create the table
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

    // 2. Use an Appender for high-speed inserts
    Appender appender(new_connection, "main", centroid_table_name); // <-- CORRECT
    
    // 'result_data.centers' is a flat vector: [c1_f1, c1_f2, ..., c2_f1, c2_f2, ...]
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
        appender.Append(Value::INTEGER(c)); // cluster_id
        appender.Append(std::move(centroid_array)); // centroid
        appender.EndRow();
    }
    
    // 4. Close the appender to flush changes
    appender.Close();
    
    printf("Centroids persisted successfully.\n");
   

    // Set the result to null
    result.SetVectorType(VectorType::CONSTANT_VECTOR);
    ConstantVector::SetNull(result, true);
}

} // namespace duckdb