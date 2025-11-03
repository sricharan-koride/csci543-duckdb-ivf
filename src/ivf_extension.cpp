#define DUCKDB_EXTENSION_MAIN

#include "ivf_extension.hpp"
#include "ivf.hpp"
#include "duckdb.hpp"
#include "duckdb/function/scalar_function.hpp"



// OpenSSL linked through vcpkg
#include <openssl/opensslv.h>

namespace duckdb {


static void LoadInternal(ExtensionLoader &loader) {
	// Register a scalar function
	auto create_ivf_index_func = ScalarFunction("create_ivf_index",
	                                        {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
	                                        LogicalType::SQLNULL, CreateIVFIndex);

	loader.RegisterFunction(create_ivf_index_func);
	
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
