#pragma once

#include "duckdb.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"

namespace duckdb {

class IVFIndexOptimizer {
public:
    // Static function matching the Optimizer callback signature
    static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan);
};

}