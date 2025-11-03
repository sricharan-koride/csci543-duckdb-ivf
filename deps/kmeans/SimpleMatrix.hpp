#ifndef KMEANS_SIMPLE_MATRIX_HPP
#define KMEANS_SIMPLE_MATRIX_HPP

#include <vector>
#include <cstddef>

namespace kmeans {

template<typename T, typename IDX>
struct SimpleMatrix {
    SimpleMatrix(IDX d, IDX n, const T* p) : nrow(d), ncol(n), data(p) {}
    IDX nrow, ncol;
    const T* data;
};

}

#endif