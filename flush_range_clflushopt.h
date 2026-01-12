#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include "parlay/parallel.h"

// Flush [ptr, ptr+bytes) from cache: write back (if dirty) + invalidate.
// Uses clflushopt when available; falls back to clflush.
// NOTE: This is expensive for multi-GB ranges; use only when you truly need it.
static inline void flush_range_clflushopt(void* ptr, size_t bytes) {
  if (ptr == nullptr || bytes == 0) return;

  constexpr size_t CL = 64;
  uintptr_t begin = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t end   = begin + bytes;

  // Align to cache-line boundaries
  uintptr_t a0 = begin & ~(uintptr_t)(CL - 1);
  uintptr_t a1 = (end + (CL - 1)) & ~(uintptr_t)(CL - 1);

  size_t lines = (a1 - a0) / CL;

  // Parallelize: each worker flushes a disjoint set of cache lines.
  parlay::parallel_for(0, lines, [&](size_t i) {
    char* line = reinterpret_cast<char*>(a0 + i * CL);
#if defined(__CLFLUSHOPT__)
    _mm_clflushopt(line);
#else
    _mm_clflush(line);
#endif
  });

  // Ensure all clflushopt requests are globally visible / completed before proceeding.
  _mm_sfence();
}
