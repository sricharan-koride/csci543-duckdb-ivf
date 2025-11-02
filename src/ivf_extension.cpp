#define DUCKDB_EXTENSION_MAIN

#include "ivf_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/scalar_function.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

// OpenSSL linked through vcpkg
#include <openssl/opensslv.h>

namespace duckdb {

inline void IvfScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &name_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(name_vector, result, args.size(), [&](string_t name) {
		return StringVector::AddString(result, "Ivf " + name.GetString() + " 🐥");
	});
}

inline void IvfOpenSSLVersionScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &name_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(name_vector, result, args.size(), [&](string_t name) {
		return StringVector::AddString(result, "Ivf " + name.GetString() + ", my linked OpenSSL version is " +
		                                           OPENSSL_VERSION_TEXT);
	});
}

static void LoadInternal(ExtensionLoader &loader) {
	// Register a scalar function
	auto ivf_scalar_function = ScalarFunction("ivf", {LogicalType::VARCHAR}, LogicalType::VARCHAR, IvfScalarFun);
	loader.RegisterFunction(ivf_scalar_function);

	// Register another scalar function
	auto ivf_openssl_version_scalar_function = ScalarFunction("ivf_openssl_version", {LogicalType::VARCHAR},
	                                                            LogicalType::VARCHAR, IvfOpenSSLVersionScalarFun);
	loader.RegisterFunction(ivf_openssl_version_scalar_function);
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
