#pragma once
#include <vector>
#include <string>
#include "duckdb.hpp"

namespace duckdb {

enum class DistanceMetric {
    L2,
    COSINE_SIM,
    INNER_PROD
};

DistanceMetric GetMetricType(const std::string &str);

float L2SquaredDistance(const std::vector<float> &a, const std::vector<float> &b);
float CosineSimilarity(const std::vector<float> &a, const std::vector<float> &b);
float InnerProduct(const std::vector<float> &a, const std::vector<float> &b);
void NormalizeVector(std::vector<float> &v);

}
