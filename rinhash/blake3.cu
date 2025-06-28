#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "blake3_device.cuh"

// Host IV (same as device)
const uint32_t IV_HOST[8] = {
    0x6A09E667UL, 0xBB67AE85UL, 0x3C6EF372UL, 0xA54FF53AUL,
    0x510E527FUL, 0x9B05688CUL, 0x1F83D9ABUL, 0x5BE0CD19UL
};

// Load words from a key (matches C implementation)
inline void load_key_words(const uint8_t key[BLAKE3_KEY_LEN], uint32_t key_words[8]) {
    for (int i = 0; i < 8; i++) {
        key_words[i] = 
            ((uint32_t)key[4 * i + 0] << 0) |
            ((uint32_t)key[4 * i + 1] << 8) |
            ((uint32_t)key[4 * i + 2] << 16) |
            ((uint32_t)key[4 * i + 3] << 24);
    }
}

// Round down to power of 2 (for tree building, matches C implementation)
inline size_t round_down_to_power_of_2(size_t x) {
    size_t ret = 1;
    while (ret < x) {
        ret *= 2;
    }
    ret /= 2;
    return ret;
}

// Count the number of 1 bits (for stack merging, matches C implementation)
inline size_t popcnt(uint64_t x) {
    size_t count = 0;
    while (x != 0) {
        count += (x & 1);
        x >>= 1;
    }
    return count;
}

// Kernel to process a single chunk
__global__ void blake3_hash_chunk_kernel(
    const uint8_t* input,
    size_t input_len,
    uint32_t* output,
    const uint32_t* key,
    uint64_t chunk_counter,
    uint8_t flags) {
    
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0) return; // Only one thread processes this chunk
    
    uint32_t cv[8];
    for (int i = 0; i < 8; i++) {
        cv[i] = key[i];
    }
    
    // Process all blocks in this chunk
    size_t offset = 0;
    uint8_t block_flags = flags;
    
    while (offset < input_len) {
        // Determine if this is the first block (CHUNK_START)
        if (offset == 0) {
            block_flags |= CHUNK_START;
        } else {
            block_flags &= ~CHUNK_START;
        }
        
        // Determine if this is the last block (CHUNK_END)
        size_t block_len = BLAKE3_BLOCK_LEN;
        if (offset + block_len >= input_len) {
            block_len = input_len - offset;
            block_flags |= CHUNK_END;
        }
        
        // Compress this block
        blake3_compress_in_place(cv, input + offset, block_len, chunk_counter, block_flags);
        offset += block_len;
    }
    
    // Output the final chaining value
    for (int i = 0; i < 8; i++) {
        output[i] = cv[i];
    }
}

// Kernel for creating a parent node
__global__ void blake3_parent_node_kernel(
    const uint8_t* left_child_cv,
    const uint8_t* right_child_cv,
    uint8_t* parent_cv,
    const uint32_t* key,
    uint8_t flags) {
    
    // Only one thread performs this operation
    if (threadIdx.x > 0 || blockIdx.x > 0) return;
    
    // Create a block containing the two child CVs
    uint8_t block[BLAKE3_BLOCK_LEN];
    for (int i = 0; i < BLAKE3_OUT_LEN; i++) {
        block[i] = left_child_cv[i];
        if (right_child_cv) {
            block[i + BLAKE3_OUT_LEN] = right_child_cv[i];
        } else {
            block[i + BLAKE3_OUT_LEN] = 0; // Padding if no right child
        }
    }
    
    // Initialize with key
    uint32_t cv[8];
    for (int i = 0; i < 8; i++) {
        cv[i] = key[i];
    }
    
    // Compress the parent node
    blake3_compress_in_place(cv, block, BLAKE3_BLOCK_LEN, 0, flags | PARENT);
    
    // Store the parent CV
    for (int i = 0; i < 8; i++) {
        ((uint32_t*)parent_cv)[i] = cv[i];
    }
}

// Host function to hash data with BLAKE3
extern "C" void blake3_hash_cuda(
    const uint8_t* input,
    size_t input_len,
    uint8_t* output) {
    
    // Skip empty input case
    if (input_len == 0) {
        for (int i = 0; i < BLAKE3_OUT_LEN; i++) {
            output[i] = 0;
        }
        return;
    }
    
    // Allocate device memory
    uint8_t* d_input;
    cudaMalloc(&d_input, input_len);
    cudaMemcpy(d_input, input, input_len, cudaMemcpyHostToDevice);
    
    // Key (IV for unkeyed hashing)
    uint32_t h_key[8];
    for (int i = 0; i < 8; i++) {
        h_key[i] = IV_HOST[i];
    }
    
    uint32_t* d_key;
    cudaMalloc(&d_key, 8 * sizeof(uint32_t));
    cudaMemcpy(d_key, h_key, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Single chunk case (most common for small inputs)
    if (input_len <= BLAKE3_CHUNK_LEN) {
        uint32_t* d_output;
        cudaMalloc(&d_output, 8 * sizeof(uint32_t));
        
        // Process the single chunk
        blake3_hash_chunk_kernel<<<1, 1>>>(
            d_input,
            input_len,
            d_output,
            d_key,
            0, // counter starts at 0
            ROOT // This is the root chunk
        );
        
        // Copy result back
        uint32_t h_result[8];
        cudaMemcpy(h_result, d_output, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        
        // Convert to bytes
        for (int i = 0; i < 8; i++) {
            output[i*4+0] = (uint8_t)(h_result[i] >> 0);
            output[i*4+1] = (uint8_t)(h_result[i] >> 8);
            output[i*4+2] = (uint8_t)(h_result[i] >> 16);
            output[i*4+3] = (uint8_t)(h_result[i] >> 24);
        }
        
        // Clean up
        cudaFree(d_output);
    } else {
        // For multi-chunk inputs, we need to implement the tree algorithm
        // This is a simplified version that doesn't handle all cases
        
        // Calculate number of chunks and allocate memory for CVs
        size_t num_chunks = (input_len + BLAKE3_CHUNK_LEN - 1) / BLAKE3_CHUNK_LEN;
        uint8_t* d_cvs;
        cudaMalloc(&d_cvs, num_chunks * BLAKE3_OUT_LEN);
        
        // Process each chunk in parallel (ideally)
        // For simplicity, we'll just use a loop here
        for (size_t i = 0; i < num_chunks; i++) {
            uint32_t* d_chunk_output;
            cudaMalloc(&d_chunk_output, 8 * sizeof(uint32_t));
            
            size_t chunk_start = i * BLAKE3_CHUNK_LEN;
            size_t chunk_len = BLAKE3_CHUNK_LEN;
            if (chunk_start + chunk_len > input_len) {
                chunk_len = input_len - chunk_start;
            }
            
            blake3_hash_chunk_kernel<<<1, 1>>>(
                d_input + chunk_start,
                chunk_len,
                d_chunk_output,
                d_key,
                i, // chunk counter
                0  // not root yet
            );
            
            // Copy to CVs array
            cudaMemcpy(d_cvs + i * BLAKE3_OUT_LEN, d_chunk_output, BLAKE3_OUT_LEN, cudaMemcpyDeviceToDevice);
            cudaFree(d_chunk_output);
        }
        
        // Now build the tree by creating parent nodes
        while (num_chunks > 1) {
            size_t new_num_chunks = (num_chunks + 1) / 2;
            uint8_t* d_parent_cvs;
            cudaMalloc(&d_parent_cvs, new_num_chunks * BLAKE3_OUT_LEN);
            
            for (size_t i = 0; i < new_num_chunks; i++) {
                uint8_t* left_cv = d_cvs + (i * 2) * BLAKE3_OUT_LEN;
                uint8_t* right_cv = (i * 2 + 1 < num_chunks) ? d_cvs + (i * 2 + 1) * BLAKE3_OUT_LEN : NULL;
                uint8_t* parent_cv = d_parent_cvs + i * BLAKE3_OUT_LEN;
                
                blake3_parent_node_kernel<<<1, 1>>>(
                    left_cv,
                    right_cv,
                    parent_cv,
                    d_key,
                    (num_chunks == 2) ? ROOT : 0 // Set ROOT for the final merge
                );
            }
            
            cudaFree(d_cvs);
            d_cvs = d_parent_cvs;
            num_chunks = new_num_chunks;
        }
        
        // Copy final result to output
        cudaMemcpy(output, d_cvs, BLAKE3_OUT_LEN, cudaMemcpyDeviceToHost);
        cudaFree(d_cvs);
    }
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_key);
}

// Wrapper function with simpler interface
extern "C" void blake3_hash(const uint8_t* input, size_t input_len, uint8_t* output) {
    blake3_hash_cuda(input, input_len, output);
}
