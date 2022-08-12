#pragma once

#include <random>
#include <stdexcept>
#include <vector>

double GetGflops(double execute_time, int M, int N, int K) {
  double numops = 1e-9 * 2 * M * N * K;
  return numops / execute_time;
}

#define abs(x) ((x) < 0.0 ? -(x) : (x))

void RandomMatrix(int m, int n, int lda, double *a) {
  std::mt19937 gen;
  std::uniform_real_distribution<double> u(-2, 2);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      a[j * lda + i] = u(gen);
    }
  }
}

void CopyMatrix(int m, int n, double *a, int lda, double *b, int ldb) {
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      b[j * ldb + i] = a[j * lda + i];
    }
  }
}

void RefMatmul(int m, int n, int k, double *a, int lda, double *b, int ldb,
               double *c, int ldc) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int p = 0; p < k; ++p) {
        c[j * ldc + i] = c[j * ldc + i] + a[p * lda + i] * b[p * ldb + j];
      }
    }
  }
}

double Compare(int m, int n, double *a, int lda, double *b, int ldb) {
  int i, j;
  double max_diff = 0.0, diff;

  for (j = 0; j < n; j++)
    for (i = 0; i < m; i++) {
      diff = abs(a[j * lda + i] - b[j * ldb + i]);
      max_diff = (diff > max_diff ? diff : max_diff);
    }

  return max_diff;
}
