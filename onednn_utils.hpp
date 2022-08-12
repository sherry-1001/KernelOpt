#pragma once

#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

inline dnnl::memory::dim product(const dnnl::memory::dims &dims) {
  return std::accumulate(dims.begin(), dims.end(), (dnnl::memory::dim)1,
                         std::multiplies<dnnl::memory::dim>());
}

// Read from handle, write to memory
inline void write_to_dnnl_memory(void *handle, dnnl::memory &mem) {
  assert(handle && "handle is nullptr.");
  dnnl::engine eng = mem.get_engine();
  size_t size = mem.get_desc().get_size();
  if (eng.get_kind() == dnnl::engine::kind::cpu) {
    uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
    assert(dst && "get_data_handle returned nullptr.");
    for (size_t i = 0; i < size; ++i)
      dst[i] = ((uint8_t *)handle)[i];
    return;
  }
}

// Read from memory, write to handle
inline void read_from_dnnl_memory(void *handle, dnnl::memory &mem) {
  assert(handle && "handle is nullptr.");
  dnnl::engine eng = mem.get_engine();
  size_t size = mem.get_desc().get_size();

  if (eng.get_kind() == dnnl::engine::kind::cpu) {
    uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
    if (!src)
      throw std::runtime_error("get_data_handle returned nullptr.");
    for (size_t i = 0; i < size; ++i)
      ((uint8_t *)handle)[i] = src[i];
    return;
  }
}