#pragma once

#include "duckdb.hpp"

namespace duckdb {

// This is the declaration for your new C++ function
// It matches the ScalarFunction signature
void CreateIVFIndex(DataChunk &args, ExpressionState &state, Vector &result);

} // namespace duckdb