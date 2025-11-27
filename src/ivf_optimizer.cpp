#include "ivf_optimizer.hpp"
#include "ivf_search.hpp" // <-- CRITICAL: Need this to access IVFSearchFunctionData
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"

namespace duckdb {

// Helper to extract IDs from an OR tree (id=1 OR id=2 OR ...)
bool ExtractIDs(Expression &expr, vector<Value> &ids) {
    if (expr.type == ExpressionType::CONJUNCTION_OR) {
        auto &conj = expr.Cast<BoundConjunctionExpression>();
        for (auto &child : conj.children) {
            if (!ExtractIDs(*child, ids)) return false;
        }
        return true;
    }

    if (expr.type == ExpressionType::COMPARE_EQUAL) {
        auto &comp = expr.Cast<BoundComparisonExpression>();
        Expression* left = comp.left.get();
        Expression* right = comp.right.get();

        if (left->type == ExpressionType::OPERATOR_CAST) {
            left = left->Cast<BoundCastExpression>().child.get();
        }

        if (left->type != ExpressionType::BOUND_COLUMN_REF) return false;
        if (right->type != ExpressionType::VALUE_CONSTANT) return false;
        
        ids.push_back(right->Cast<BoundConstantExpression>().value);
        return true;
    }

    if (expr.type == ExpressionType::COMPARE_IN) {
        auto &op = expr.Cast<BoundOperatorExpression>();
        Expression* left = op.children[0].get();
        
        if (left->type == ExpressionType::OPERATOR_CAST) {
            left = left->Cast<BoundCastExpression>().child.get();
        }
        
        if (left->type != ExpressionType::BOUND_COLUMN_REF) return false;

        for (size_t i = 1; i < op.children.size(); i++) {
            if (op.children[i]->type != ExpressionType::VALUE_CONSTANT) return false;
            ids.push_back(op.children[i]->Cast<BoundConstantExpression>().value);
        }
        return true;
    }

    return false;
}

void IVFIndexOptimizer::Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
    for (auto &child : plan->children) {
        Optimize(input, child);
    }

    if (plan->type != LogicalOperatorType::LOGICAL_FILTER) return;
    if (plan->children.size() != 1) return;
    
    auto &child = plan->children[0];
    if (child->type != LogicalOperatorType::LOGICAL_GET) return;

    auto &table_function = child->Cast<LogicalGet>();
    if (table_function.function.name != "ann_search") return;

    auto &filter = plan->Cast<LogicalFilter>();
    
    for (size_t i = 0; i < filter.expressions.size(); i++) {
        auto &expr = filter.expressions[i];
        
        vector<Value> allowed_ids;
        if (ExtractIDs(*expr, allowed_ids)) {
            
            printf("🚀 IVF OPTIMIZER: Pushdown detected! Moving %zu IDs into ann_search.\n", allowed_ids.size());
            
            // 1. Update named_parameters (For consistency/Explain)
            table_function.named_parameters["allowed_ids"] = Value::LIST(LogicalType::BIGINT, allowed_ids);
            
            // 2. CRITICAL FIX: Update the Bind Data directly!
            // Since Bind has already run, we must update the struct that Init will read.
            if (table_function.bind_data) {
                auto &bind_data = table_function.bind_data->Cast<IVFSearchFunctionData>();
                bind_data.has_filter = true;
                bind_data.allowed_ids.clear();
                for (const auto &val : allowed_ids) {
                    bind_data.allowed_ids.insert(val.GetValue<int64_t>());
                }
            }

            // Remove the expression from the filter
            filter.expressions.erase(filter.expressions.begin() + i);
            
            if (filter.expressions.empty()) {
                plan = std::move(child);
            }
            return;
        }
    }
}

} // namespace duckdb