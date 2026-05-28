// BF16 <-> FP32 conversion utilities.
//
// Truncation (not rounding): float_to_bf16 drops the lower 16 bits of
// the FP32 mantissa, matching the hardware behavior of v_cvt_pk_bf16_f32.
// These functions are usable on both host and device (__host__ __device__).

#pragma once
#include <cstdint>
#include <cstring>

#ifdef __HIPCC__
#define BF16_HOST_DEVICE __host__ __device__
#else
#define BF16_HOST_DEVICE
#endif

// Convert FP32 to BF16 by truncating the lower 16 mantissa bits.
inline BF16_HOST_DEVICE uint16_t float_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}

// Convert BF16 to FP32 by zero-extending the lower 16 bits.
inline BF16_HOST_DEVICE float bf16_to_float(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
