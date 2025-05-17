#ifndef BLAKE2B_CUDA_CUH
#define BLAKE2B_CUDA_CUH

#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ inline uint64_t rotr64_dev(uint64_t x, uint32_t n) {
    return (x >> n) | (x << (64 - n));
}

// IV values defined in the spec
__constant__ uint64_t blake2b_IV_cuda[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

__device__ void blake2b_hash(const uint8_t* input, size_t input_len, uint8_t* output) {
    // State h and local block buffer
    uint64_t h[8];
    uint64_t m[16];

    // Initialization
    for (int i = 0; i < 8; i++) h[i] = blake2b_IV_cuda[i];
    h[0] ^= 0x01010020;  // digest_length=32, key_length=0, fanout=1, depth=1

    // Zero block padding
    uint8_t block[128] = {0};
    for (size_t i = 0; i < input_len; i++) block[i] = input[i];

    // Load block into m[0..15] in little-endian
    for (int i = 0; i < 16; i++) {
        m[i] = ((uint64_t*)block)[i];
    }

    // Compress (single block)
    uint64_t v[16];
    for (int i = 0; i < 8; i++) v[i] = h[i];
    for (int i = 0; i < 8; i++) v[i + 8] = blake2b_IV_cuda[i];

    v[12] ^= 128;           // t0 = 128 (input length)
    v[14] ^= 0xFFFFFFFFFFFFFFFFULL; // f0 = 0xFF..

    // 12 rounds (simplified with constant sigma)
    #pragma unroll
    for (int round = 0; round < 12; round++) {
        // Message mixing (simplified sigma = identity)
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int a = i;
            int b = (i + 4) % 8;
            int c = (i + 8) % 16;
            int d = (i + 12) % 16;

            v[a] = v[a] + v[b] + m[(2 * i) % 16];
            v[d] = rotr64_dev(v[d] ^ v[a], 32);
            v[c] = v[c] + v[d];
            v[b] = rotr64_dev(v[b] ^ v[c], 24);
            v[a] = v[a] + v[b] + m[(2 * i + 1) % 16];
            v[d] = rotr64_dev(v[d] ^ v[a], 16);
            v[c] = v[c] + v[d];
            v[b] = rotr64_dev(v[b] ^ v[c], 63);
        }
    }

    for (int i = 0; i < 8; i++) {
        h[i] ^= v[i] ^ v[i + 8];
    }

    for (int i = 0; i < 4; i++) { // output 32 bytes
        ((uint64_t*)output)[i] = h[i];
    }
}

#endif
