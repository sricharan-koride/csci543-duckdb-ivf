#ifndef KMEANS_KMEANS_HPP
#define KMEANS_KMEANS_HPP

#include "SimpleMatrix.hpp"
#include <vector>
#include <random>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace kmeans {

// --- MOVED INTERNAL NAMESPACE TO THE TOP ---
namespace internal {

inline std::mt19937& get_PRNG() {
    static thread_local std::mt19937 prng(std::random_device{}());
    return prng;
}

template<typename T>
T squared_dist(const T* x, const T* y, size_t n) {
    T dist = 0;
    for (size_t i = 0; i < n; ++i) {
        T diff = x[i] - y[i];
        dist += diff * diff;
    }
    return dist;
}

template<typename T, typename CLUSTER>
CLUSTER find_nearest_center(const T* vec, size_t ndim, const T* centers, CLUSTER ncenters) {
    CLUSTER best_c = 0;
    T min_dist = squared_dist(vec, centers, ndim);
    for (CLUSTER c = 1; c < ncenters; ++c) {
        T dist = squared_dist(vec, centers + c * ndim, ndim);
        if (dist < min_dist) {
            min_dist = dist;
            best_c = c;
        }
    }
    return best_c;
}

template<typename T, typename IDX, typename CLUSTER>
void compute_clusters(const SimpleMatrix<T, IDX>& data, const T* centers, CLUSTER ncenters, std::vector<CLUSTER>& clusters) {
    for (IDX i = 0; i < data.ncol; ++i) {
        clusters[i] = find_nearest_center(data.data + i * data.nrow, data.nrow, centers, ncenters);
    }
}

template<typename T, typename IDX, typename CLUSTER>
void compute_centroids(const SimpleMatrix<T, IDX>& data, const CLUSTER* clusters, CLUSTER ncenters, T* new_centers, IDX* counts) {
    for (IDX i = 0; i < data.ncol; ++i) {
        CLUSTER clust = clusters[i];
        if (clust < 0 || clust >= ncenters) continue; 
        counts[clust]++;
        auto cptr = new_centers + clust * data.nrow;
        auto dptr = data.data + i * data.nrow;
        for (IDX r = 0; r < data.nrow; ++r) {
            cptr[r] += dptr[r];
        }
    }
}

template<typename T>
bool check_convergence(const T* old_centers, const T* new_centers, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (old_centers[i] != new_centers[i]) return false;
    }
    return true;
}

template<typename T, typename IDX>
void fill_centers_from_indices(const SimpleMatrix<T, IDX>& data, const std::vector<IDX>& indices, T* center_ptr) {
    for (size_t i = 0; i < indices.size(); ++i) {
        std::copy(data.data + indices[i] * data.nrow, data.data + (indices[i] + 1) * data.nrow, center_ptr + i * data.nrow);
    }
}

template<typename T, typename IDX, typename CLUSTER>
void update_min_dist(const SimpleMatrix<T, IDX>& data, const T* centers, CLUSTER ncenters, std::vector<T>& min_dist) {
    const T* center_ptr = centers + (ncenters - 1) * data.nrow;
    for (IDX i = 0; i < data.ncol; ++i) {
        T dist = squared_dist(data.data + i * data.nrow, center_ptr, data.nrow);
        if (ncenters == 1 || dist < min_dist[i]) {
            min_dist[i] = dist;
        }
    }
}

} // namespace internal
// --- END OF INTERNAL NAMESPACE ---


template<typename T, typename IDX, typename CLUSTER>
struct InitializeRandom {
    void run(const SimpleMatrix<T, IDX>& data, CLUSTER ncenters, T* center_ptr, std::vector<IDX>& indices) const {
        auto& reng = internal::get_PRNG();
        std::uniform_int_distribution<IDX> dist(0, data.ncol - 1);
        for (CLUSTER c = 0; c < ncenters; ++c) {
            indices[c] = dist(reng);
        }
        internal::fill_centers_from_indices(data, indices, center_ptr);
        return;
    }
};

template<typename T, typename IDX, typename CLUSTER>
struct InitializeKmeanspp {
    void run(const SimpleMatrix<T, IDX>& data, CLUSTER ncenters, T* center_ptr, std::vector<IDX>& indices) const {
        auto& reng = internal::get_PRNG();
        std::uniform_int_distribution<IDX> dist(0, data.ncol - 1);
        indices[0] = dist(reng);

        std::vector<T> min_dist(data.ncol);
        internal::fill_centers_from_indices(data, indices, center_ptr);
        T* center_copy = center_ptr;

        for (CLUSTER c = 1; c < ncenters; ++c) {
            internal::update_min_dist(data, center_copy, c, min_dist);
            std::discrete_distribution<IDX> weighted_dist(min_dist.begin(), min_dist.end());
            indices[c] = weighted_dist(reng);
            internal::fill_centers_from_indices(data, indices, center_ptr);
        }
        return;
    }
};

template<typename T, typename IDX>
struct RefineLloydOptions {
    int max_iterations = 100;
};

template<typename T, typename IDX, typename CLUSTER>
struct RefineLloyd {
    RefineLloydOptions<T, IDX> options;
    RefineLloyd(RefineLloydOptions<T, IDX> o) : options(o) {}
    RefineLloyd() = default;

    struct Results {
        Results(CLUSTER nc, IDX nr) : centers(nc * nr), clusters(0), iterations(0) {}
        std::vector<T> centers;
        std::vector<CLUSTER> clusters;
        int iterations;
    };

    Results run(const SimpleMatrix<T, IDX>& data, CLUSTER ncenters, T* center_ptr) const {
        Results output(ncenters, data.nrow);
        std::vector<CLUSTER> clusters(data.ncol);
        std::vector<T> new_centers(data.nrow * ncenters);
        std::vector<IDX> counts(ncenters);

        int iter = 0;
        bool changed = true;
        while (iter < options.max_iterations && changed) {
            changed = false;
            std::fill(new_centers.begin(), new_centers.end(), 0);
            std::fill(counts.begin(), counts.end(), 0);

            internal::compute_clusters(data, center_ptr, ncenters, clusters);
            internal::compute_centroids(data, clusters.data(), ncenters, new_centers.data(), counts.data());

            for (CLUSTER c = 0; c < ncenters; ++c) {
                if (counts[c]) {
                    auto cptr = new_centers.data() + c * data.nrow;
                    for (IDX r = 0; r < data.nrow; ++r) {
                        cptr[r] /= counts[c];
                    }
                }
            }
            
            if (internal::check_convergence(center_ptr, new_centers.data(), new_centers.size())) {
                changed = false;
            } else {
                std::copy(new_centers.begin(), new_centers.end(), center_ptr);
                changed = true;
            }
            ++iter;
        }

        std::copy(center_ptr, center_ptr + new_centers.size(), output.centers.begin());
        output.clusters = std::move(clusters);
        output.iterations = iter;
        return output;
    }
};

// --- FIX 2: Added explicit C++11 return type ---
template<typename T, typename IDX, typename CLUSTER, class INIT, class REFINE>
typename REFINE::Results compute(const SimpleMatrix<T, IDX>& data, const INIT& init, CLUSTER ncenters, const REFINE& refine, int nthreads) {
    std::vector<T> centers(data.nrow * ncenters);
    std::vector<IDX> indices(ncenters);
    init.run(data, ncenters, centers.data(), indices);
    return refine.run(data, ncenters, centers.data());
}

} // namespace kmeans

#endif