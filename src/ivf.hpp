#pragma once

#include "duckdb.hpp"

namespace duckdb {

void CreateIVFIndex(DataChunk &args, ExpressionState &state, Vector &result);

} 