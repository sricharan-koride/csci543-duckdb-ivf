#include "metric.hpp"
#include <cmath>
#include <stdexcept>
#include "duckdb/common/string_util.hpp"

namespace duckdb {

DistanceMetric GetMetricType(const std::string &str) {
    auto s = StringUtil::Lower(str);
    if (s == "l2" || s == "euclidean") return DistanceMetric::L2;
    if (s == "cosine" || s == "cosine_similarity") return DistanceMetric::COSINE_SIM;
    if (s == "inner_product" || s == "dot_product" || s == "ip") return DistanceMetric::INNER_PROD;
    return DistanceMetric::L2;
}

float L2SquaredDistance(const std::vector<float> &a, const std::vector<float> &b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

float InnerProduct(const std::vector<float> &a, const std::vector<float> &b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

float CosineSimilarity(const std::vector<float> &a, const std::vector<float> &b) {
    // Robust implementation that computes norms
    float dot = 0.0f;
    float norm_a_sq = 0.0f;
    float norm_b_sq = 0.0f;
    for(size_t i=0; i<a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a_sq += a[i] * a[i];
        norm_b_sq += b[i] * b[i];
    }
    float denom = std::sqrt(norm_a_sq) * std::sqrt(norm_b_sq);
    if (denom < 1e-9) return 0.0f;
    return dot / denom;
}

void NormalizeVector(std::vector<float> &v) {
    float sum_sq = 0.0f;
    for (float x : v) sum_sq += x * x;
    float norm = std::sqrt(sum_sq);
    if (norm > 1e-9) {
        for (float &x : v) x /= norm;
    }
}

}
