#include <iostream>
#include<cuda_runtime.h>
#include <chrono>


// I hope to use this main file for all of the reduction files
int main(){
    int n = 1<<20; // about 1M elements for now
    size_t bytes = n * sizeof(int);

    // Host/CPU arrays
    int *host_input_data = new int[n];  
    int *host_output_data = new int[n + 255] / 256; // to have sufficient size for output array
    
    // Device/GPU arrays
    int *dev_input_data, *dev_output_data;

    //Init data
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
    num_blocks = (n + blockSize - 1) / blockSize;
    num_threads = blockSize * sizeof(int);
    reduce0<<<num_blocks, num_threads>>>(dev_input_data, dev_output_data);
    cudaDeviceSynchronize();

    auto stop = std::chorno::high_resolution_clock::now();
    auto duration - std::chrono::duration_cast<std::milliseconds>(stop - start).count();

    // Copying data back to the host (CPU)
    cudaMemcpy(host_output_data, dev_output_data, (n + 255) / 256 * sizeof(int), cudaMemcpyDeviceToHost);

    // Final reduction on the host
    int finalResult = h_odata[0];
    for (int i = 1; i < (n + 255) / 256; ++i) {
        finalResult = min(finalResult, h_odata[i]);
    }

    std::cout << "Reduced result: " << finalResult << std::endl;
    std::cout << "Time elapsed: " << duration << " ms" << std::endl;

    // Computing bandwidth
    double bandwisth = bytes / duration / 1e6; // computed in GB/s
    std::cout << "Effective bandwidth: " << bandwidth << " GB/s" << std:endl;

    // Freeing memory
    cudaFree(dev_input_data);
    cudaFree(dev_output_data);
    delete [] host_input_data;
    delete [] host_output_data;
}