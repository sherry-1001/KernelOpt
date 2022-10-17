#include <iostream>

#include "allocator.hpp"
#include "config.hpp"
#include "opt_gemm.hpp"
#include "utils.hpp"

using namespace std;

int main() {
  int m = 1024, n = 1024, k = 1024, rep;
  double best_cost_time = 0;
  double *a, *b, *c, *bias, *refc;

  a = (double *)cpu_allocator_t::malloc(m * k * sizeof(double), 32);
  b = (double *)cpu_allocator_t::malloc(k * n * sizeof(double), 32);
  c = (double *)cpu_allocator_t::malloc(m * n * sizeof(double), 32);
  bias = (double *)cpu_allocator_t::malloc(m * n * sizeof(double), 32);
  refc = (double *)cpu_allocator_t::malloc(m * n * sizeof(double), 32);

  ConstantMatrix(m, k, a, 1);
  ConstantMatrix(k, n, b, 2);
  ZeroMatrix(m, n, bias);

  CopyMatrix(m, n, bias, m, refc, m);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int p = 0; p < k; ++p) {
        refc[i * n + j] += a[i * k + p] * b[p * n + j];
      }
    }
  }

  for (rep = 0; rep < NREPEATS; rep++) {
    CopyMatrix(m, n, bias, c);
    // Execute Start Timer
    auto start_time = dclock();

    OptMM1(m, n, k, a, b, c);

    // Execute End Timer
    auto duration = dclock() - start_time;
    if (rep == 0) {
      best_cost_time = duration;
    } else {
      best_cost_time = duration < best_cost_time ? duration : best_cost_time;
    }
  }

  double diff = Compare(m, n, c, refc);
  double gflops = GetGflops(best_cost_time, m, n, k);

  printf("%d %le %le\n", m, gflops, diff);
  fflush(stdout);
  return 0;
}