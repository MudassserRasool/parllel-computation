// file: cpu_vs_gpu.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__ void vectorAddGPU(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

void vectorAddCPU(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++) C[i] = A[i] + B[i];
}

int main() {
    int N = 1 << 24; // ~16 million elements
    size_t size = N * sizeof(float);

    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // CPU timing
    auto start = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_A, h_B, h_C, N);
    auto end = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "CPU Time: " << cpuTime << " ms\n";

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // GPU timing
    start = std::chrono::high_resolution_clock::now();
    vectorAddGPU<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double gpuTime = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "GPU Time: " << gpuTime << " ms\n";

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Show speedup
    if (gpuTime > 0)
        std::cout << "GPU is about " << cpuTime / gpuTime << "x faster than CPU\n";

    // Cleanup
    delete[] h_A; delete[] h_B; delete[] h_C;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
