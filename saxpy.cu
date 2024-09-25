
#include <stdio.h>
#include <cuda_runtime.h>

/**
 * 
 * saxpy    : A * X[i] + Y[i]
 * dtype    : float32
 * input    : X(vector), Y(vector)
 * param    : A(constant)
 * 
 */


// -- CPU --

void saxpy_cpu(float a, float *x, float *y, int n) {
    for(int i = 0; i < n; i++)
        y[i] = a * x[i] + y[i];
}

void print_vec(float* x, int n) {
    for(int i = 0; i < n; i++) {
        if(i == (n-1)) {
            printf("%.4f\n", x[i]);
        } else {
            printf("%.4f, ", x[i]);
        }
    }
}

// -- CUDA --

__global__ void saxpy_cuda(float a, float *x, float *y, int n) {
    int t_id = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = t_id; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}


int main() {
    int n = 3;
    int nb = n * sizeof(float);
    float a = 2.0f;
    float x[] = {6, 4, 2};
    float y[] = {2, 3, 4};
    // saxpy_cpu(a, x, y, n);
    print_vec(y, n);


    // cuda
    float *x_d, *y_d;
    cudaMalloc(&x_d, nb);
    cudaMalloc(&y_d, nb);
    cudaMemcpy(x_d, x, nb, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, nb, cudaMemcpyHostToDevice);
    saxpy_cuda<<<32, 1023>>>(a, x_d, y_d, n);
    cudaDeviceSynchronize();
    cudaMemcpy(y, y_d, nb, cudaMemcpyDeviceToHost);
    print_vec(y, n);
    cudaFree(x_d);
    cudaFree(y_d);

    return 0;
}