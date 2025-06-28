#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <stdexcept>

// Include shared device functions
#include "rinhash_device.cuh"
#include "argon2d_device.cuh"
#include "sha3-256.cu"
#include "blake3_device.cuh"


// External references to our CUDA implementations
extern "C" void blake3_hash(const uint8_t* input, size_t input_len, uint8_t* output);
extern "C" void argon2d_hash_rinhash(uint8_t* output, const uint8_t* input, size_t input_len);
extern "C" void sha3_256_hash(const uint8_t* input, size_t input_len, uint8_t* output);

// Modified kernel to use device functions
extern "C" __global__ void rinhash_cuda_kernel(
    const uint8_t* input, 
    size_t input_len, 
    uint8_t* output
) {
    // Intermediate results in shared memory
    uint8_t blake3_out[32];
    uint8_t argon2_out[32];
    
    // Only one thread should do this work
    if (threadIdx.x == 0) {
        // Step 1: BLAKE3 hash - now using light_hash_device
        light_hash_device(input, input_len, blake3_out);
        // Step 2: Argon2d hash
        uint32_t m_cost = 64; // Example
        size_t memory_size = m_cost * sizeof(block);
        block* d_memory = (block*)malloc(memory_size);
        uint8_t salt[11] = { 'R','i','n','C','o','i','n','S','a','l','t' };
        device_argon2d_hash(argon2_out, blake3_out, 32, 2, 64, 1, d_memory, salt, 11);
        
        // Step 3: SHA3-256 hash
        uint8_t sha3_out[32];
        sha3_256_device(argon2_out, 32, sha3_out);
        
    }
    
    // Use syncthreads to ensure all threads wait for the computation to complete
    __syncthreads();
}

// Modified kernel to use device functions
extern "C" __global__ void rinhash_cuda_kernel_batch(
    const uint8_t* headers,         // num_blocks * 80 bytes
    size_t header_len,              // = 80
    uint8_t* outputs,               // num_blocks * 32 bytes
    uint32_t num_blocks
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_blocks) {
        return;
    }
    const uint8_t* input = headers + tid * header_len;
    uint8_t* output = outputs + tid * 32;

    // RinHash Steps
    uint8_t blake3_out[32];
    light_hash_device(input, header_len, blake3_out);

    uint32_t m_cost = 64;
    block* memory = (block*)malloc(m_cost * sizeof(block));
    if (!memory) return;

    uint8_t salt[11] = { 'R','i','n','C','o','i','n','S','a','l','t' };
    uint8_t argon2_out[32];
    device_argon2d_hash(argon2_out, blake3_out, 32, 2, m_cost, 1, memory, salt, sizeof(salt));

    sha3_256_device(argon2_out, 32, output);
    free(memory);
}


// RinHash CUDA implementation
extern "C" void rinhash_cuda(const uint8_t* input, size_t input_len, uint8_t* output) {
    // Allocate device memory
    uint8_t *d_input = nullptr;
    uint8_t *d_output = nullptr;

    cudaError_t err;

    // Allocate memory on device
    err = cudaMalloc(&d_input, input_len);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to allocate input memory: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMalloc(&d_output, 32);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to allocate output memory: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        return;
    }
    

    // Copy input data to device
    err = cudaMemcpy(d_input, input, input_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to copy input to device: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }

    // Launch the kernel
    rinhash_cuda_kernel<<<1, 1>>>(d_input, input_len, d_output);

    // Wait for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error during kernel execution: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }

    // Copy result back to host
    err = cudaMemcpy(output, d_output, 32, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to copy output from device: %s\n", cudaGetErrorString(err));
    }
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}


// Helper function to convert a block header to bytes
extern "C" void blockheader_to_bytes(
    const uint32_t* version,
    const uint32_t* prev_block,
    const uint32_t* merkle_root,
    const uint32_t* timestamp,
    const uint32_t* bits,
    const uint32_t* nonce,
    uint8_t* output,
    size_t* output_len
) {
    size_t offset = 0;
    
    // Version (4 bytes)
    memcpy(output + offset, version, 4);
    offset += 4;
    
    // Previous block hash (32 bytes)
    memcpy(output + offset, prev_block, 32);
    offset += 32;
    
    // Merkle root (32 bytes)
    memcpy(output + offset, merkle_root, 32);
    offset += 32;
    
    // Timestamp (4 bytes)
    memcpy(output + offset, timestamp, 4);
    offset += 4;
    
    // Bits (4 bytes)
    memcpy(output + offset, bits, 4);
    offset += 4;
    
    // Nonce (4 bytes)
    memcpy(output + offset, nonce, 4);
    offset += 4;
    
    *output_len = offset;
}

// Batch processing version for mining
extern "C" void rinhash_cuda_batch(
    const uint8_t* block_headers,
    size_t block_header_len,
    uint8_t* outputs,
    uint32_t num_blocks
) {
    uint8_t *d_headers = nullptr, *d_outputs = nullptr;
    size_t headers_size = block_header_len * num_blocks;
    size_t outputs_size = 32 * num_blocks;
    cudaError_t err;

    cudaMalloc(&d_headers, headers_size);
    cudaMalloc(&d_outputs, outputs_size);
    cudaMemset(d_outputs, 0xee, outputs_size);

    cudaMemcpy(d_headers, block_headers, headers_size, cudaMemcpyHostToDevice);

    // <<<block数, スレッド数>>>
    const int threads_per_block = 128;
    int blocks = (num_blocks + threads_per_block - 1) / threads_per_block;
    rinhash_cuda_kernel_batch<<<blocks, threads_per_block>>>(
        d_headers, block_header_len, d_outputs, num_blocks
    );

    cudaDeviceSynchronize();

    err = cudaMemcpy(outputs, d_outputs, outputs_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to copy output from device: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_headers);
    cudaFree(d_outputs);
}


// Main RinHash function that would be called from outside
extern "C" void RinHash(
    const uint32_t* version,
    const uint32_t* prev_block,
    const uint32_t* merkle_root,
    const uint32_t* timestamp,
    const uint32_t* bits,
    const uint32_t* nonce,
    uint8_t* output
) {
    uint8_t block_header[80]; // Standard block header size
    size_t block_header_len;
    
    // Convert block header to bytes
    blockheader_to_bytes(
        version,
        prev_block,
        merkle_root,
        timestamp,
        bits,
        nonce,
        block_header,
        &block_header_len
    );
    
    // Calculate RinHash
    rinhash_cuda(block_header, block_header_len, output);
}

bool is_better(uint8_t* hash1, uint8_t* hash2) {
    for (int i = 7; i >= 0; i--) {
        uint32_t h1 = ((uint32_t)hash1[i*4 + 0]) |
                      ((uint32_t)hash1[i*4 + 1] << 8) |
                      ((uint32_t)hash1[i*4 + 2] << 16) |
                      ((uint32_t)hash1[i*4 + 3] << 24);

        uint32_t h2 = ((uint32_t)hash2[i*4 + 0]) |
                      ((uint32_t)hash2[i*4 + 1] << 8) |
                      ((uint32_t)hash2[i*4 + 2] << 16) |
                      ((uint32_t)hash2[i*4 + 3] << 24);

        if (h1 < h2) return true;
        if (h1 > h2) return false;
    }
    return false; // equal
}

// Mining function that tries different nonces
extern "C" void RinHash_mine(
    const uint32_t* work_data,
    uint32_t nonce_offset,
    uint32_t start_nonce,
    uint32_t num_nonces,
    uint32_t* found_nonce,
    uint8_t* target_hash,
    uint8_t* best_hash
) {
    const size_t block_header_len = 80;
    int headerbytes = block_header_len * num_nonces;
    int hashbytes = 32 * num_nonces;
    uint8_t block_headers[80 * 1024];
    uint8_t hashes[32 * 1024];
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256 * 1024 * 1024); // 128MB
    // Prepare block headers with different nonces
    for (uint32_t i = 0; i < num_nonces; i++) {
        uint32_t current_nonce = start_nonce + i;
        uint32_t work_data_copy[20];
        
        // Copy work data and update nonce
        memcpy(work_data_copy, work_data, 80);
        work_data_copy[nonce_offset] = current_nonce;
        
        // Fill header
        uint8_t* header = block_headers + i * block_header_len;
        size_t header_len;
        
        memcpy(header, work_data_copy, 80);
    }
    
    // Calculate hashes for all nonces
    rinhash_cuda_batch(block_headers, block_header_len, hashes, num_nonces);
    
    // Initialize best_hash with maximum value (worst possible hash)
    memcpy(best_hash, hashes, 32); // Initialize to the first hash
    *found_nonce = start_nonce;
    
    // Find the best hash
    for (uint32_t i = 0; i < num_nonces; i++) {
        uint8_t* current_hash = hashes + i * 32;
        
        bool is_current_better = false;

        
        if (is_better(current_hash, best_hash)) {
            memcpy(best_hash, current_hash, 32);
            *found_nonce = start_nonce + i;
        }
    }
}
