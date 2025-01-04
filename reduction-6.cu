#include <iostream>
#include<cuda_runtime.h>
#include <chrono>
#include <numeric> 

// Adding this function to help with unrolling and adding the Template
template <unsigned int blockSize>
__device__ void warpReduce(volatile int* sdata, unsigned int tid){
    if(blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if(blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if(blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if(blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if(blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if(blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

// REDUCTION 6 – Multiple Adds / Threads
template <int blockSize>
__global__ void reduce6(int *g_in_data, int *g_out_data, unsigned int n){
    extern __shared__ int sdata[];  // stored in the shared memory

    // Each thread loading one element from global onto shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;
    sdata[tid] = 0;

    while(i < n) { 
      sdata[tid] += g_in_data[i] + g_in_data[i + blockSize]; 
      i += gridSize; 
    }
    __syncthreads();

    // Perform reductions in steps, reducing thread synchronization
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    }

    if (tid < 32) warpReduce<blockSize>(sdata, tid);

    if (tid == 0){
        g_out_data[blockIdx.x] = sdata[0];
    }
}

// I hope to use this main file for all of the reduction files
int main(){
    int n = 1<<22; // Increase to about 4M elements
    size_t bytes = n * sizeof(int);

    // Host/CPU arrays
    int *host_input_data = new int[n];
    int *host_output_data = new int[(n + 255) / 256]; // to have sufficient size for output array

    // Device/GPU arrays
    int *dev_input_data, *dev_output_data;

    // Init data
    srand(42); // Fixed seed
    for (int i = 0; i < n; i++){
        host_input_data[i] = rand() % 100;
    }

    // Allocating memory on GPU for device arrays
    cudaMalloc(&dev_input_data, bytes);
    cudaMalloc(&dev_output_data, (n + 255) / 256 * sizeof(int));

    // Copying our data onto the device (GPU)
    cudaMemcpy(dev_input_data, host_input_data, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256; // number of threads per block
    int num_blocks = (n + (2 * blockSize) - 1) / (2 * blockSize);   // Modifying this to account for the fact that 1 thread accesses 2 elements

    auto start = std::chrono::high_resolution_clock::now(); // start timer

    // Needed for Complete unrolling
    // Launch Kernel and Synchronize threads
    switch (blockSize) {
        case 512:
            reduce6<512><<<num_blocks, 512, 512 * sizeof(int)>>>(dev_input_data, dev_output_data, n);
            break;
        case 256:
            reduce6<256><<<num_blocks, 256, 256 * sizeof(int)>>>(dev_input_data, dev_output_data, n);
            break;
        case 128:
            reduce6<128><<<num_blocks, 128, 128 * sizeof(int)>>>(dev_input_data, dev_output_data, n);
            break;
    }
    
    cudaDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0; // duration in milliseconds with three decimal points

    // Copying data back to the host (CPU)
    cudaMemcpy(host_output_data, dev_output_data, (n + 255) / 256 * sizeof(int), cudaMemcpyDeviceToHost);

    // Final reduction on the host
    int finalResult = host_output_data[0];
    for (int i = 1; i < (n + 255) / 256; ++i) {
        finalResult += host_output_data[i];
    }

    // CPU Summation for verification
    int cpuResult = std::accumulate(host_input_data, host_input_data + n, 0);
    if (cpuResult == finalResult) {
        std::cout << "\033[32m"; // Set text color to green
        std::cout << "Verification successful: GPU result matches CPU result.\n";
        std::cout << "GPU Result: " << finalResult << ", CPU Result: " << cpuResult << std::endl;
    } else {
        std::cout << "\033[31m"; // Set text color to red
        std::cout << "Verification failed: GPU result (" << finalResult << ") does not match CPU result (" << cpuResult << ").\n";
        std::cout << "GPU Result: " << finalResult << ", CPU Result: " << cpuResult << std::endl;
    }
    std::cout << "\033[0m"; // Reset text color to default

    double bandwidth = (duration > 0) ? (bytes / duration / 1e6) : 0; // computed in GB/s, handling zero duration
    std::cout << "Reduced result: " << finalResult << std::endl;
    std::cout << "Time elapsed: " << duration << " ms" << std::endl;
    std::cout << "Effective bandwidth: " << bandwidth << " GB/s" << std::endl;

    // Freeing memory
    cudaFree(dev_input_data);
    cudaFree(dev_output_data);
    delete[] host_input_data;
    delete[] host_output_data;
}
