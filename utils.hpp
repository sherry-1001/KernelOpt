#pragma once

#include <random>
#include <stdexcept>
#include <vector>

#include <sys/time.h>
#include <time.h>

static double gtod_ref_time_sec = 0.0;

double dclock() {
  double the_time, norm_sec;
  struct timeval tv;

  gettimeofday(&tv, NULL);

  if (gtod_ref_time_sec == 0.0)
    gtod_ref_time_sec = (double)tv.tv_sec;

  norm_sec = (double)tv.tv_sec - gtod_ref_time_sec;

  the_time = norm_sec + tv.tv_usec * 1.0e-6;

  return the_time;
}

double GetGflops(double execute_time, int M, int N, int K) {
  double numops = 1.0e-09 * 2.0 * M * N * K;
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

void ConstantMatrix(int row, int col, double *a, double const_value) {
  for (int r = 0; r < row; ++r) {
    for (int c = 0; c < col; ++c) {
      a[r * col + c] = const_value;
    }
  }
}

void ZeroMatrix(int row, int col, double *a) {
  for (int r = 0; r < row; ++r) {
    for (int c = 0; c < col; ++c) {
      a[r * col + c] = 0;
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

void CopyMatrix(int row, int col, double *a, double *b) {
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j) {
      b[i * col + j] = a[i * col + j];
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

double Compare(int row, int col, double *a, double *b) {
  int i, j;
  double max_diff = 0.0, diff;

  for (i = 0; i < row; i++)
    for (j = 0; j < col; j++) {
      diff = abs(a[i * col + j] - b[i * col + j]);
      max_diff = (diff > max_diff ? diff : max_diff);
    }
  return max_diff;
}