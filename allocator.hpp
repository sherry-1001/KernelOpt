#pragma once

#include <memory>

/// Default allocator for CPU
class cpu_allocator_t {
public:
  constexpr static size_t DEFAULT_ALIGNMENT = 64;

  static void *malloc(size_t size, size_t alignment) {
    void *ptr = nullptr;
    const size_t align = alignment == 0 ? DEFAULT_ALIGNMENT : alignment;
#ifdef _WIN32
    ptr = _aligned_malloc(size, align);
    int rc = ((ptr) ? 0 : errno);
#else
    int rc = ::posix_memalign(&ptr, align, size);
#endif /* _WIN32 */
    return (rc == 0) ? ptr : nullptr;
  }

  static void free(void *p) {
#ifdef _WIN32
    _aligned_free((void *)p);
#else
    ::free((void *)p);
#endif /* _WIN32 */
  }
};
