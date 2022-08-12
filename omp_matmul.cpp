#include <algorithm>
#include <assert.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>

#include "utils.hpp"
using namespace std;

void OmpMatmul(const std::vector<std::vector<double>> &A,
               const std::vector<std::vector<double>> &B,
               std::vector<std::vector<double>> &C) {
  int M = A.size();
  int K = A[0].size();
  int N = B[0].size();
  assert(K == (int)B.size() && "matirx A and matrix B K axis shoule be equal.");

#pragma omp parallel for private(i, j, k) shared(A, B, C)
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

int main() { // init
  int M = 1024, N = 1024, K = 1024;
  std::vector<std::vector<double>> A, B, C;

  // init
  A.resize(M, vector<double>(K));
  B.resize(N, vector<double>(K));
  C.resize(M, vector<double>(N));
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        A[i][k] = (double)rand() / (double)RAND_MAX;
        B[k][j] = (double)rand() / (double)RAND_MAX;
      }
      C[i][j] = 0;
    }
  }

  std::cout << "===============start===============\n";
  auto start_time = std::chrono::system_clock::now();
  OmpMatmul(A, B, C);
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now() - start_time);

  double gflops = GetGflops(duration.count(), M, N, K);

  std::cout << "naive matmul gflops: " << gflops << "\n";
  return 0;
}
