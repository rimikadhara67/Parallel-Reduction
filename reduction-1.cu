#include <iostream>
#include<cuda_runtime.h>
#include <chrono>
#include <algorithm>

// REDUCTION 1 – Interleaved Addressing without branch divergence
__global__ void reduce1(int *g_in_data, int *g_out_data){
    extern __shared__ int sdata[];  // stored in the shared memory

    // Each thread loading one element from global onto shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_in_data[i];
    __syncthreads();

    // Reduction method -- occurs in shared memory
    for(unsigned int s = 1; s < blockDim.x; s *= 2){
        // note the stride as s *= 2 : this causes the interleaving addressing
        int index = 2 * s * tid;    // now we don't need a diverging branch from the if condition
        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];   // s is used to denote the offset that will be combined
        }
        __syncthreads();
    }
    if (tid == 0){
        g_out_data[blockIdx.x] = sdata[0];
    }
}

// I hope to use this main file for all of the reduction files
int main(){
    int n = 1 << 20; // about 1M elements for now
    size_t bytes = n * sizeof(int);

    // Host/CPU arrays
    int *host_input_data = new int[n];  
    int *host_output_data = new int[(n + 255) / 256]; // to have sufficient size for output array
    
    // Device/GPU arrays
    int *dev_input_data, *dev_output_data;

    //Init data
    srand(42); // Fixed seed
    for (int i = 0; i < n; i++){
        host_input_data[i] = rand() % 100;
    }

    // Allocating memory on GPU for device arrays
    cudaMalloc(&dev_input_data, bytes);
    cudaMalloc(&dev_output_data, (n + 255) / 256 * sizeof(int));

    // Copying our data onto the device (GPU)
    cudaMemcpy(dev_input_data, host_input_data, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256; //number of threads per block

    auto start = std::chrono::high_resolution_clock::now(); //start timer

    // Launch Kernel and Synchronize threads
    int num_blocks = (n + blockSize - 1) / blockSize;
    int num_threads = blockSize * sizeof(int);
    reduce1<<<num_blocks, num_threads>>>(dev_input_data, dev_output_data);
    cudaDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

    // Copying data back to the host (CPU)
    cudaMemcpy(host_output_data, dev_output_data, (n + 255) / 256 * sizeof(int), cudaMemcpyDeviceToHost);

    // Final reduction on the host
    int finalResult = host_output_data[0];
    for (int i = 1; i < (n + 255) / 256; ++i) {
        finalResult = min(finalResult, host_output_data[i]);
    }

    std::cout << "Reduced result: " << finalResult << std::endl;
    std::cout << "Time elapsed: " << duration << " ms" << std::endl;

    int cpuResult = *std::min_element(host_input_data, host_input_data + n);
    if (cpuResult == finalResult) {
        std::cout << "Verification successful: GPU result matches CPU result." << std::endl;
        std::cout << "GPU Result: " << finalResult << ", CPU Result: " << cpuResult << std::endl;
    } else {
        std::cout << "Verification failed: GPU result (" << finalResult << ") does not match CPU result (" << cpuResult << ")." << std::endl;
        std::cout << "GPU Result: " << finalResult << ", CPU Result: " << cpuResult << std::endl;
    }

    // Computing bandwidth
    double bandwidth = bytes / duration / 1e6; // computed in GB/s
    std::cout << "Effective bandwidth: " << bandwidth << " GB/s" << std::endl;

    // Freeing memory
    cudaFree(dev_input_data);
    cudaFree(dev_output_data);
    delete [] host_input_data;
    delete [] host_output_data;
}