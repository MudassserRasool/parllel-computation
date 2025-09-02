#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// ---------------- CPU Rendering Simulation ----------------
void renderCPU(const std::vector<float>& input, std::vector<float>& output) {
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < input.size(); i++) {
        // Expensive operation simulation (like shading or pixel ops)
        output[i] = input[i] * input[i] + 0.5f;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "CPU Render Time: " << duration.count() << " ms\n";
}

// ---------------- GPU Rendering Simulation ----------------
__global__ void renderGPUKernel(const float* d_input, float* d_output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_output[idx] = d_input[idx] * d_input[idx] + 0.5f;
    }
}

void renderGPU(const std::vector<float>& input, std::vector<float>& output) {
    int n = input.size();
    size_t size = n * sizeof(float);

    float *d_input, *d_output;

    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    auto start = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d_input, input.data(), size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    renderGPUKernel<<<numBlocks, blockSize>>>(d_input, d_output, n);

    cudaMemcpy(output.data(), d_output, size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "GPU Render Time: " << duration.count() << " ms\n";

    cudaFree(d_input);
    cudaFree(d_output);
}

// ---------------- Main ----------------
int main() {
    const int N = 1 << 20; // ~1 million pixels
    std::vector<float> input(N, 0.5f);
    std::vector<float> outputCPU(N), outputGPU(N);

    renderCPU(input, outputCPU);
    renderGPU(input, outputGPU);

    // Verify correctness
    for (int i = 0; i < 5; i++) {
        std::cout << "CPU[" << i << "]=" << outputCPU[i]
                  << " | GPU[" << i << "]=" << outputGPU[i] << "\n";
    }

    return 0;
}
