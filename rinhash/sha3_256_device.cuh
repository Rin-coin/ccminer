#ifndef SHA3_256_DEVICE_CUH
#define SHA3_256_DEVICE_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

// SHA3-256 constants
#define SHA3_256_RATE 136
#define SHA3_256_CAPACITY 64
#define SHA3_256_HASH_LEN 32
#define SHA3_KECCAK_ROUNDS 24

// SHA3 state
typedef struct {
    uint64_t state[25];
} sha3_state;

// Rotation constants
__device__ __constant__ uint8_t SHA3_ROTC[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

// Permutation indices
__device__ __constant__ uint8_t SHA3_PILN[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

// Rotate left 64-bit value
__device__ __forceinline__ uint64_t sha3_rotl64(uint64_t x, uint8_t n) {
    return (x << n) | (x >> (64 - n));
}

// Keccak-f[1600] permutation function
__device__ void sha3_keccakf(uint64_t st[25]) {
    uint64_t t, bc[5];
    
    // 24 rounds of permutation
    for (int r = 0; r < SHA3_KECCAK_ROUNDS; r++) {
        
        // Theta step
        for (int i = 0; i < 5; i++) {
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        }
        
        for (int i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ sha3_rotl64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5) {
                st[j + i] ^= t;
            }
        }
        
        // Rho Pi
        t = st[1];
        for (int i = 0; i < 24; i++) {
            uint8_t j = SHA3_PILN[i];
            bc[0] = st[j];
            st[j] = sha3_rotl64(t, SHA3_ROTC[i]);
            t = bc[0];
        }
        
        // Chi
        for (int j = 0; j < 25; j += 5) {
            for (int i = 0; i < 5; i++) {
                bc[i] = st[j + i];
            }
            for (int i = 0; i < 5; i++) {
                st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
            }
        }
        
        // Iota
        // Simplified round constants
        uint64_t keccak_round_constants[24] = {
            0x0000000000000001ULL, 0x0000000000008082ULL,
            0x800000000000808aULL, 0x8000000080008000ULL,
            0x000000000000808bULL, 0x0000000080000001ULL,
            0x8000000080008081ULL, 0x8000000000008009ULL,
            0x000000000000008aULL, 0x0000000000000088ULL,
            0x0000000080008009ULL, 0x000000008000000aULL,
            0x000000008000808bULL, 0x800000000000008bULL,
            0x8000000000008089ULL, 0x8000000000008003ULL,
            0x8000000000008002ULL, 0x8000000000000080ULL,
            0x000000000000800aULL, 0x800000008000000aULL,
            0x8000000080008081ULL, 0x8000000000008080ULL,
            0x0000000080000001ULL, 0x8000000080008008ULL
        };
        st[0] ^= keccak_round_constants[r];
    }
}

// Initialize SHA3 state
__device__ void sha3_init(sha3_state* s) {
    // Zero out the state
    for (int i = 0; i < 25; i++) {
        s->state[i] = 0;
    }
}

// Update SHA3 state with input data
__device__ void sha3_update(sha3_state* s, const uint8_t* in, size_t inlen) {
    size_t i;
    uint8_t* state = (uint8_t*)s->state;
    
    // XOR input into state
    for (i = 0; i < inlen; i++) {
        state[i % SHA3_256_RATE] ^= in[i];
        
        if ((i + 1) % SHA3_256_RATE == 0) {
            sha3_keccakf(s->state);
        }
    }
}

// Finalize SHA3 hash
__device__ void sha3_final(sha3_state* s, uint8_t* md) {
    uint8_t* state = (uint8_t*)s->state;
    
    // Padding: add 0x06 (domain separator for SHA3-256) followed by 10*1
    state[0] ^= 0x06;
    state[SHA3_256_RATE - 1] ^= 0x80;
    
    // Final permutation
    sha3_keccakf(s->state);
    
    // Extract hash
    for (int i = 0; i < SHA3_256_HASH_LEN; i++) {
        md[i] = state[i];
    }
}

// Complete SHA3-256 function
__device__ void sha3_256_device(const void* in, size_t inlen, void* md) {
    sha3_state s;
    sha3_init(&s);
    sha3_update(&s, (const uint8_t*)in, inlen);
    sha3_final(&s, (uint8_t*)md);
}

#endif // SHA3_256_DEVICE_CUH
