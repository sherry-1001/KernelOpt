#include <cassert>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "onednn_utils.hpp"
#include "utils.hpp"

#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

void init_vector(std::vector<float> &v) {
  std::mt19937 gen;
  std::uniform_real_distribution<float> u(-1, 1);

  for (auto &e : v)
    e = u(gen);
}

double matmul_example(dnnl::engine::kind engine_kind) {

  // Create execution dnnl::engine.
  dnnl::engine engine(engine_kind, 0);

  // Create dnnl::stream.
  dnnl::stream engine_stream(engine);

  // Tensor dimensions.
  const memory::dim M = 1024, K = 512, N = 2048;

  // Source (src), weights, and destination (dst) tensors dimensions.
  memory::dims src_dims = {M, K};
  memory::dims weights_dims = {K, N};
  memory::dims dst_dims = {M, N};

  // Allocate buffers.
  std::vector<float> src_data(M * K), weights_data(K * N), dst_data(M * N);

  // Initialize src, weights, bias.
  init_vector(src_data);
  init_vector(weights_data);

  // Create memory descriptors and memory objects for src, weights, and
  // dst.
  auto src_md = memory::desc(src_dims, dt::f32, tag::ab);
  auto weights_md = memory::desc(weights_dims, dt::f32, tag::ab);
  auto dst_md = memory::desc(dst_dims, dt::f32, tag::ab);

  auto src_mem = memory(src_md, engine);
  auto weights_mem = memory(weights_md, engine);
  auto dst_mem = memory(dst_md, engine);

  // Write data to memory object's handles.
  write_to_dnnl_memory(src_data.data(), src_mem);
  write_to_dnnl_memory(weights_data.data(), weights_mem);

  // Create operation descriptor
  auto matmul_d = matmul::desc(src_md, weights_md, dst_md);

  // Create primitive descriptor. without any attrs
  auto matmul_pd = matmul::primitive_desc(matmul_d, engine);

  // Create the primitive.
  auto matmul_prim = matmul(matmul_pd);

  // Primitive arguments.
  std::unordered_map<int, memory> matmul_args;
  matmul_args.insert({DNNL_ARG_SRC, src_mem});
  matmul_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
  matmul_args.insert({DNNL_ARG_DST, dst_mem});

  // Execute Start Timer
  auto start_time = std::chrono::system_clock::now();

  // Primitive execution: matrix multiplication
  matmul_prim.execute(engine_stream, matmul_args);

  // Wait for the computation to finalize.
  engine_stream.wait();

  // Read data from memory object's handle.
  read_from_dnnl_memory(dst_data.data(), dst_mem);

  // Execute End Timer
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now() - start_time);

  double gflops = GetGflops(duration.count(), M, N, K);
  return gflops;
}

int main() {

  std::cout << matmul_example(dnnl::engine::kind::cpu) << "\n";
  return 0;
}