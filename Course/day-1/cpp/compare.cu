#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// ---------------- GPU Kernel ----------------
__global__ void addGPU(int *a, int *b, int *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// ---------------- CPU Function ----------------
void addCPU(int *a, int *b, int *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(int);

    // host arrays
    int *h_a = new int[N];
    int *h_b = new int[N];
    int *h_c = new int[N];
    int *h_c_cpu = new int[N];

    // initialize data
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // device arrays
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // copy to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // ---------------- GPU timing ----------------
    auto startGPU = std::chrono::high_resolution_clock::now();

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addGPU<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    auto endGPU = std::chrono::high_resolution_clock::now();

    // ---------------- CPU timing ----------------
    auto startCPU = std::chrono::high_resolution_clock::now();
    addCPU(h_a, h_b, h_c_cpu, N);
    auto endCPU = std::chrono::high_resolution_clock::now();

    // compute times
    auto gpuTime = std::chrono::duration<double, std::milli>(endGPU - startGPU).count();
    auto cpuTime = std::chrono::duration<double, std::milli>(endCPU - startCPU).count();

    std::cout << "CPU Time: " << cpuTime << " ms\n";
    std::cout << "GPU Time: " << gpuTime << " ms\n";
    std::cout << "Speedup: " << cpuTime / gpuTime << "x\n";

    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_c_cpu;

    return 0;
}
