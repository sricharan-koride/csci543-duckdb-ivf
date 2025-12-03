#include "pq.hpp"
#include "duckdb.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

using namespace duckdb;


// Utility: L2 distance

static float L2Sq(const float *a, const float *b, int dim) {
    float acc = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        acc += diff * diff;
    }
    return acc;
}


// PQ Training (k-means per subspace)

void TrainPQCodebooks(const std::vector<std::vector<float>> &vectors,
                      PQMetadata meta,
                      PQCodebook &out) {

    int dim = meta.dim;
    int M = meta.M;
    int Ks = meta.Ks;

    if (vectors.empty())
        throw std::runtime_error("PQ training: input vectors empty.");

    if (dim % M != 0)
        throw std::runtime_error("PQ training: dim % M != 0.");

    int subdim = dim / M;

    out.M = M;
    out.Ks = Ks;
    out.subvector_dim = subdim;
    out.codebooks.resize(M);

    // Flatten all subvectors for each subspace
    for (int m = 0; m < M; m++) {
        // Extract subspace slices
        std::vector<std::vector<float>> subspace_data;
        subspace_data.reserve(vectors.size());

        for (auto &v : vectors) {
            std::vector<float> slice(subdim);
            std::copy(v.begin() + m * subdim, v.begin() + (m + 1) * subdim, slice.begin());
            subspace_data.push_back(slice);
        }

        // Naive k-means++ init: pick first point
        std::vector<std::vector<float>> centroids(Ks, std::vector<float>(subdim, 0.0f));
        centroids[0] = subspace_data[0];

        // Random-ish initialization: copy first Ks entries if available
        for (int k = 1; k < Ks && k < (int)subspace_data.size(); k++) {
            centroids[k] = subspace_data[k];
        }

        // Lloyd iterations
        for (int iter = 0; iter < 15; iter++) {
            std::vector<std::vector<float>> new_c(Ks, std::vector<float>(subdim, 0.0f));
            std::vector<int> counts(Ks, 0);

            for (auto &vec : subspace_data) {
                // find nearest centroid
                float best = 1e30f;
                int best_k = 0;
                for (int k = 0; k < Ks; k++) {
                    float dist = L2Sq(vec.data(), centroids[k].data(), subdim);
                    if (dist < best) {
                        best = dist;
                        best_k = k;
                    }
                }

                for (int d = 0; d < subdim; d++)
                    new_c[best_k][d] += vec[d];
                counts[best_k]++;
            }

            for (int k = 0; k < Ks; k++) {
                if (counts[k] > 0) {
                    for (int d = 0; d < subdim; d++)
                        new_c[k][d] /= counts[k];
                } else {
                    // fallback
                    new_c[k] = centroids[k];
                }
            }

            centroids = std::move(new_c);
        }

        // flatten centroids
        std::vector<float> flat(Ks * subdim);
        for (int k = 0; k < Ks; k++) {
            for (int d = 0; d < subdim; d++) {
                flat[k * subdim + d] = centroids[k][d];
            }
        }
        out.codebooks[m] = std::move(flat);
    }
}


// PQ Encoding

void EncodePQCodes(const std::vector<std::vector<float>> &vectors,
                   const PQCodebook &codebook,
                   std::vector<std::vector<uint8_t>> &out) {

    int M = codebook.M;
    int Ks = codebook.Ks;
    int subdim = codebook.subvector_dim;

    out.resize(vectors.size(), std::vector<uint8_t>(M));

    for (size_t i = 0; i < vectors.size(); i++) {
        const auto &v = vectors[i];
        for (int m = 0; m < M; m++) {
            const float *cb = codebook.codebooks[m].data(); 
            const float *slice = v.data() + m * subdim;

            float best = 1e30f;
            uint8_t best_k = 0;

            for (int k = 0; k < Ks; k++) {
                float dist = L2Sq(slice, cb + k * subdim, subdim);
                if (dist < best) {
                    best = dist;
                    best_k = (uint8_t)k;
                }
            }

            out[i][m] = best_k;
        }
    }
}


// Build Distance LUT

void BuildPQDistanceLUT(const std::vector<float> &query,
                        const PQCodebook &codebook,
                        std::vector<std::vector<float>> &lut) {

    int M = codebook.M;
    int Ks = codebook.Ks;
    int subdim = codebook.subvector_dim;

    lut.resize(M, std::vector<float>(Ks, 0.0f));

    for (int m = 0; m < M; m++) {
        const float *cb = codebook.codebooks[m].data();
        const float *slice = query.data() + m * subdim;

        for (int k = 0; k < Ks; k++) {
            lut[m][k] = L2Sq(slice, cb + k * subdim, subdim);
        }
    }
}


// PQ Distance

float PQDistance(const std::vector<uint8_t> &pq_code,
                 const std::vector<std::vector<float>> &lut) {

    float dist = 0.0f;
    int M = pq_code.size();
    for (int m = 0; m < M; m++) {
        dist += lut[m][pq_code[m]];
    }
    return dist;
}


// Persistence: Codebooks

void PersistPQCodebooks(ClientContext &context,
                        const string &index_name,
                        const PQCodebook &codebook) {

    string tbl = "ivf_pq_codebooks_" + index_name;

    // Create table if not exists
    {
        Connection c(DatabaseInstance::GetDatabase(context));
        c.Query("CREATE TABLE IF NOT EXISTS " + tbl +
                "(m INTEGER, k INTEGER, values FLOAT[]);");
    }

    {
        Connection c(DatabaseInstance::GetDatabase(context));
        auto prep = c.Prepare("INSERT INTO " + tbl + " (m, k, values) VALUES ($1, $2, $3);");
        for (int m = 0; m < codebook.M; m++) {
            const auto &flat = codebook.codebooks[m];
            int subdim = codebook.subvector_dim;
            duckdb::vector<Value> list_vals;
            for (int k = 0; k < codebook.Ks; k++) {
                list_vals.reserve(subdim);
                for (int d = 0; d < subdim; d++) {
                    list_vals.emplace_back(Value(flat[k * subdim + d]));
                }
                prep->Execute(m, k, Value::LIST(list_vals));
                list_vals.clear();
            }
        }
    }
}


// Persistence: Codes

void PersistPQCodes(ClientContext &context,
                    const string &index_name,
                    const std::vector<std::vector<uint8_t>> &codes) {

    string tbl = "ivf_pq_codes_" + index_name;

    {
        Connection c(DatabaseInstance::GetDatabase(context));
        c.Query("CREATE TABLE IF NOT EXISTS " + tbl +
                "(id BIGINT, code UTINYINT[]);");
    }

    {
        Connection c(DatabaseInstance::GetDatabase(context));
        auto prep = c.Prepare("INSERT INTO " + tbl + " (id, code) VALUES ($1, $2);");
        for (size_t i = 0; i < codes.size(); i++) {
            const auto &code = codes[i];
            duckdb::vector<Value> list_vals;
            list_vals.reserve(code.size());
            for (auto b : code) {
                list_vals.emplace_back(Value((int32_t)b));
            }
            prep->Execute((int64_t)i, Value::LIST(list_vals));
        }
    }
}


// Load Codebooks

void LoadPQCodebooks(ClientContext &context,
                     const string &index_name,
                     PQCodebook &codebook) {

    string tbl = "ivf_pq_codebooks_" + index_name;

    Connection c(DatabaseInstance::GetDatabase(context));
    auto res = c.Query("SELECT m, k, values FROM " + tbl + " ORDER BY m, k;");

    if (res->HasError()) {
        throw std::runtime_error(res->GetError());
    }

    int M = 0;
    int Ks = 0;
    int subdim = 0;

    // First pass - infer M, Ks, subdim
    for (auto &row : *res) {
        int m = row.GetValue<int>(0);
        int k = row.GetValue<int>(1);

        // values is a FLOAT[] LIST
        Value list_val = row.GetValue<Value>(2);
        auto &children = ListValue::GetChildren(list_val);

        std::vector<float> arr;
        arr.reserve(children.size());
        for (auto &child : children) {
            arr.push_back(child.GetValue<float>());
        }

        M = std::max(M, m + 1);
        Ks = std::max(Ks, k + 1);
        subdim = static_cast<int>(arr.size());
    }

    codebook.M = M;
    codebook.Ks = Ks;
    codebook.subvector_dim = subdim;
    codebook.codebooks.resize(M, std::vector<float>(Ks * subdim));

    // Second pass - fill data
    for (auto &row : *res) {
        int m = row.GetValue<int>(0);
        int k = row.GetValue<int>(1);

        Value list_val = row.GetValue<Value>(2);
        auto &children = ListValue::GetChildren(list_val);

        std::vector<float> arr;
        arr.reserve(children.size());
        for (auto &child : children) {
            arr.push_back(child.GetValue<float>());
        }

        float *dest = codebook.codebooks[m].data() + k * subdim;
        std::copy(arr.begin(), arr.end(), dest);
    }
}

void LoadPQCodes(ClientContext &context,
                 const string &index_name,
                 std::vector<std::vector<uint8_t>> &codes) {

    string tbl = "ivf_pq_codes_" + index_name;

    Connection c(DatabaseInstance::GetDatabase(context));
    auto res = c.Query("SELECT id, code FROM " + tbl + " ORDER BY id;");

    if (res->HasError()) {
        throw std::runtime_error(res->GetError());
    }

    idx_t max_id = 0;
    for (auto &row : *res) {
        idx_t id = row.GetValue<idx_t>(0);
        if (id > max_id) max_id = id;
    }

    codes.clear();
    codes.resize(max_id + 1);

    for (auto &row : *res) {
        idx_t id = row.GetValue<idx_t>(0);

        Value list_val = row.GetValue<Value>(1);
        auto &children = ListValue::GetChildren(list_val);

        std::vector<uint8_t> arr;
        arr.reserve(children.size());
        for (auto &child : children) {
            arr.push_back(child.GetValue<uint8_t>());
        }

        codes[id] = std::move(arr);
    }
}