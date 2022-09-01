#pragma once

#include <algorithm>
#include <assert.h>
#include <chrono>
#include <fstream>
#include <immintrin.h>
#include <omp.h>
#include <vector>

#include "utils.hpp"

void OptMM0(int m, int n, int k, double *a, int lda, double *b, int ldb,
            double *c, int ldc) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int p = 0; p < k; p++) {
        c[j * ldc + i] = c[j * ldc + i] + a[p * lda + i] * b[p * ldb + j];
      }
    }
  }
}

void OptMM1(int m, int n, int k, double *a, double *b, double *c) {
  __m256d c_256;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j += 4) {
      c_256 = _mm256_load_pd(c + (i * n + j));
      for (int p = 0; p < k; p++) {
        __m256d a_256 = _mm256_broadcast_sd(a + i * k + p);
        __m256d b_256 = _mm256_load_pd(b + (p * n + j));
        c_256 = _mm256_fmadd_pd(a_256, b_256, c_256);
      }
      _mm256_store_pd(c + (i * n + j), c_256);
    }
  }
}

template <typename T>
inline void gemm_block_caculate(int sm, int sn, int sk, T *a, T *b, T *c, int m,
                                int n, int k, int block_size) {
  __m256d c_256;
  for (int i = sm; i < sm + block_size; i++) {
    for (int j = sn; j < sn + block_size; j += 4) {
      c_256 = _mm256_load_pd(c + (i * n + j));
      for (int p = 0; p < sk + block_size; p++) {
        __m256d a_256 = _mm256_broadcast_sd(a + i * k + p);
        __m256d b_256 = _mm256_load_pd(b + (p * n + j));
        c_256 = _mm256_fmadd_pd(a_256, b_256, c_256);
      }
      _mm256_store_pd(c + (i * n + j), c_256);
    }
  }
}

void OptMM2(int m, int n, int k, double *a, double *b, double *c) {
  int block_size = 64;
  for (int sm = 0; sm < m; sm += block_size) {
    for (int sn = 0; sn < n; sn += block_size) {
      for (int sk = 0; sk < k; sk += block_size) {
        gemm_block_caculate(sm, sn, sk, a, b, c, m, n, k, block_size);
      }
    }
  }
}
