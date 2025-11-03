#include "ivf.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"
#include <vector>
#include "duckdb/main/database.hpp"
#include "duckdb/main/connection.hpp"

namespace duckdb {

void CreateIVFIndex(DataChunk &args, ExpressionState &state, Vector &result) {
    // 1. Get the arguments
    auto index_name = args.GetValue(0, 0).ToString();
    auto table_name = args.GetValue(1, 0).ToString();
    auto column_name = args.GetValue(2, 0).ToString();
    printf("CreateIVFIndex called:\n");
    printf("  Index Name: %s\n", index_name.c_str());
    printf("  Table Name: %s\n", table_name.c_str());
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

    std::vector<std::vector<float>> sample_vectors;
    printf("Successfully fetched sample vectors. Processing...\n");

    while (auto chunk = query_result->Fetch()) {
        if (!chunk || chunk->size() == 0) {
            break;
        }

        auto &vector_column = chunk->data[0];
        auto logical_type = vector_column.GetType().id();

        if (logical_type == LogicalTypeId::LIST) {
            // --- Original LIST handling ---
            auto list_data = ListVector::GetData(vector_column);
            auto &child_vector = ListVector::GetEntry(vector_column);
            auto float_data = FlatVector::GetData<float>(child_vector);
            auto row_count = chunk->size();

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
            // --- NEW: Handle ARRAY columns ---
            auto array_type = ArrayType::GetChildType(vector_column.GetType());
            auto array_size = ArrayType::GetSize(vector_column.GetType());
            auto &child_vector = ArrayVector::GetEntry(vector_column);
            auto float_data = FlatVector::GetData<float>(child_vector);
            auto row_count = chunk->size();

            for (idx_t i = 0; i < row_count; i++) {
                std::vector<float> row_vector;
                row_vector.reserve(array_size);
                for (idx_t j = 0; j < array_size; j++) {
                    row_vector.push_back(float_data[i * array_size + j]);
                }
                sample_vectors.push_back(std::move(row_vector));
            }

        } else {
            printf("Error: Unsupported vector type %s\n",
                   LogicalTypeIdToString(logical_type).c_str());
        }
    }

    printf("Vector sampling complete. Total vectors sampled: %zu\n", sample_vectors.size());

    result.SetVectorType(VectorType::CONSTANT_VECTOR);
    ConstantVector::SetNull(result, true);
}

} // namespace duckdb
