#define DUCKDB_EXTENSION_MAIN

#include "ivf_extension.hpp"
#include "ivf.hpp"
#include "duckdb.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "ivf_search.hpp" 



// OpenSSL linked through vcpkg
#include <openssl/opensslv.h>

namespace duckdb {


static void LoadInternal(ExtensionLoader &loader) {
	// Register a scalar function
	auto create_ivf_index_func = ScalarFunction("create_ivf_index",
	                                        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
	                                        LogicalType::SQLNULL, CreateIVFIndex);

	TableFunction ann_search_func("ann_search", {LogicalType:: VARCHAR, LogicalType::ARRAY(LogicalType::FLOAT, 128), LogicalType::INTEGER, LogicalType::INTEGER}
,ComputeIVFSearch, BindIVFSearch, InitIVFSearch);

	ann_search_func.named_parameters["nprobe"] = LogicalType::INTEGER;
	loader.RegisterFunction(create_ivf_index_func);
	loader.RegisterFunction(ann_search_func);
}

void IvfExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
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

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(ivf, loader) {
	duckdb::LoadInternal(loader);
}
}
