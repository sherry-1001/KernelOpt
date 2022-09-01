#include <iostream>

#include "config.hpp"
#include "opt_gemm.hpp"
#include "utils.hpp"

using namespace std;

int main() {
  int p, m, n, k, lda, ldb, ldc, rep;
  double best_cost_time = 0;
  int opt_level = 0;
  vector<double> a, b, c, bias, refc;

  for (p = PFIRST; p <= PLAST; p *= PINC) {
    m = (M == -1 ? p : M);
    n = (N == -1 ? p : N);
    k = (K == -1 ? p : K);

    lda = (LDA == -1 ? m : LDA);
    ldb = (LDB == -1 ? k : LDB);
    ldc = (LDC == -1 ? m : LDC);

    a.resize(lda * (k + 1));
    b.resize(ldb * n);
    c.resize(ldc * n);
    bias.resize(ldc * n);
    refc.resize(ldc * n);

    RandomMatrix(m, k, lda, &a[0]);
    RandomMatrix(k, n, ldb, &b[0]);
    RandomMatrix(m, n, ldc, &bias[0]);

    CopyMatrix(m, n, &bias[0], ldc, &refc[0], ldc);
    RefMatmul(m, n, k, &a[0], lda, &b[0], ldb, &refc[0], ldc);

    for (rep = 0; rep < NREPEATS; rep++) {
      CopyMatrix(m, n, &bias[0], ldc, &c[0], ldc);
      // Execute Start Timer
      auto start_time = dclock();

      OptMM0(m, n, k, &a[0], lda, &b[0], ldb, &c[0], ldc);

      // Execute End Timer
      auto duration = dclock() - start_time;
      if (rep == 0) {
        best_cost_time = duration;
      } else {
        best_cost_time = duration < best_cost_time ? duration : best_cost_time;
      }
    }

    double diff = Compare(m, n, &c[0], ldc, &refc[0], ldc);
    double gflops = GetGflops(best_cost_time, m, n, k);

    printf("%d %lf %lf \n", p, gflops, diff);
    fflush(stdout);
  }
  return 0;
}
