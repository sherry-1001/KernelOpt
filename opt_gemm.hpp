#pragma once

#include <algorithm>
#include <assert.h>
#include <chrono>
#include <fstream>
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

void OptLevel(int opt_level, int m, int n, int k, double *a, int lda, double *b,
              int ldb, double *c, int ldc) {
  switch (opt_level) {
  case 0:
    OptMM0(m, n, k, a, lda, b, ldb, c, ldc);
  }
}