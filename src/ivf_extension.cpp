#define DUCKDB_EXTENSION_MAIN

#include "ivf_extension.hpp"
#include "ivf_search.hpp"
#include "ivf_optimizer.hpp"
#include "ivf.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/parser/parsed_data/create_info.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

// OpenSSL linked through vcpkg
#include <openssl/opensslv.h>

namespace duckdb {

static void LoadInternal(DatabaseInstance &db) {
    auto &catalog = Catalog::GetSystemCatalog(db);
    
    // Create a connection for registration
    Connection con(db);
    
    // --- FIX: Start a Transaction ---
    // Accessing the catalog requires an active transaction context.
    con.BeginTransaction();
    
    auto &context = *con.context;

    // --- 1. Register CREATE_IVF_INDEX (Scalar Function) ---
    ScalarFunction create_ivf_index_func("create_ivf_index", 
        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
        LogicalType::VARCHAR, 
        CreateIVFIndex
    );
    
    CreateScalarFunctionInfo scalar_info(create_ivf_index_func);
    scalar_info.on_conflict = OnCreateConflict::IGNORE_ON_CONFLICT;
    catalog.CreateFunction(context, scalar_info);


    // --- 2. Register ANN_SEARCH (Table Function) ---
    // Use a variable-length list for the query vector so different dimensionalities
    // are accepted (avoids hard-coded FLOAT[128] signature mismatches).
    TableFunction ann_search_func("ann_search", 
        {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::FLOAT), LogicalType::INTEGER, LogicalType::INTEGER}, 
        ComputeIVFSearch, 
        BindIVFSearch,
        InitIVFSearch
    );
    ann_search_func.named_parameters["nprobe"] = LogicalType::INTEGER;
    ann_search_func.named_parameters["allowed_ids"] = LogicalType::LIST(LogicalType::BIGINT);
    // Named parameter that allows passing a serialized WHERE fragment from the optimizer
    // into the table-function so the inner candidate-selection SQL can apply the
    // same predicate (e.g., s.region = 'US').
    ann_search_func.named_parameters["where_clause"] = LogicalType::VARCHAR;

    CreateTableFunctionInfo table_func_info(ann_search_func);
    table_func_info.on_conflict = OnCreateConflict::IGNORE_ON_CONFLICT;
    catalog.CreateFunction(context, table_func_info);


    // --- 3. Register Optimizer ---
    OptimizerExtension ivf_optimizer;
    ivf_optimizer.optimize_function = IVFIndexOptimizer::Optimize;
    
    db.config.optimizer_extensions.push_back(ivf_optimizer);
    
    // --- FIX: Commit the Transaction ---
    con.Commit();
    
    printf("IVF Extension loaded (v1.4 compliant).\n");
}

// --- C++ Class Implementation ---
void IvfExtension::Load(ExtensionLoader &loader) {
    LoadInternal(loader.GetDatabaseInstance());
}

std::string IvfExtension::Name() {
    return "ivf";
}

std::string IvfExtension::Version() const {
#ifdef EXT_VERSION_IVF
    return EXT_VERSION_IVF;
#else
    return "";
#endif
}

} // namespace duckdb

// --- C Entry Point ---
extern "C" {

DUCKDB_EXTENSION_API void ivf_duckdb_cpp_init(duckdb::ExtensionLoader &loader) {
    duckdb::LoadInternal(loader.GetDatabaseInstance());
}

DUCKDB_EXTENSION_API const char *ivf_version() {
    return "v0.0.1";
}

}