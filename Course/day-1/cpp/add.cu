#include <iostream>
#include <cuda_runtime.h>

__global__ void add(int *a, int *b, int *c, int n) {
    int idx = threadIdx.x;   // index within the block
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 10; // length of arrays
    int h_a[N], h_b[N], h_c[N]; // host arrays

    // initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    int *d_a, *d_b, *d_c; // device pointers
    size_t size = N * sizeof(int);

    // allocate device memory
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // copy host arrays to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // launch kernel with one block of N threads
    add<<<1, N>>>(d_a, d_b, d_c, N);

    // copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // print result
    std::cout << "Result: ";
    for (int i = 0; i < N; i++) {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
