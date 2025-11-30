#pragma once
#include <vector>
#include <string>
#include "duckdb.hpp"
using std::string;
using namespace duckdb;

struct PQCodebook {
    int M;            // num subspaces
    int Ks;           // centroids per subspace (e.g., 256)
    int subvector_dim;
    std::vector<std::vector<float>> codebooks; // flattened: M codebooks each of size Ks*subvector_dim
};

struct PQMetadata {
    int dim;          // vector dimension (128)
    int M;            // num subspaces
    int Ks;           // K for codebooks
};

void TrainPQCodebooks(const std::vector<std::vector<float>> &vectors, PQMetadata meta, PQCodebook &out);
void EncodePQCodes(const std::vector<std::vector<float>> &vectors, const PQCodebook &codebook, std::vector<std::vector<uint8_t>> &out);

void BuildPQDistanceLUT(const std::vector<float> &query, const PQCodebook &codebook, std::vector<std::vector<float>> &lut);
float PQDistance(const std::vector<uint8_t> &pq_code, const std::vector<std::vector<float>> &lut);

void PersistPQCodebooks(ClientContext &context, const string &index_name, const PQCodebook &codebook);
void PersistPQCodes(ClientContext &context, const string &index_name, const std::vector<std::vector<uint8_t>> &codes);

void LoadPQCodebooks(ClientContext &context, const string &index_name, PQCodebook &codebook);
void LoadPQCodes(ClientContext &context, const string &index_name, std::vector<std::vector<uint8_t>> &codes);