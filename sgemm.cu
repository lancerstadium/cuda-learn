#include <stdio.h>
#include <cuda_runtime.h>

// ref: https://openmlsys.github.io/chapter_accelerator/accelerator_practise.html

void sfill_rand_cpu(float* A, unsigned n) {
    for(int i = 0; i < n; i++) {
        A[i] = (float)(rand() % 100) + 0;
    }
}

void scopy_cpu(float* dst, float* src, unsigned n) {
    memcpy(dst, src, n * sizeof(float));
}

void sprint_2d_cpu(float* A, unsigned m, unsigned n) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            printf("%10.4f ", A[i * n + j]);
        }
        putchar('\n');
    }
    putchar('\n');
}

void sgemm_cpu(float* A, float* B, float* C, float alpha, float beta, 
                unsigned M, unsigned N, unsigned K) {
    for(unsigned m = 0; m < M; ++m) {
        for(unsigned n = 0; n < N; ++n) {
            float c = 0;
            for(unsigned k = 0; k < K; ++k) {
                c += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = alpha * c + beta * C[m * N + n];
        }
    }
}



// O(M * N * K + 2 * M * N)
__global__ void sgemm_cuda(float* A, float* B, float* C, float alpha, float beta, 
                unsigned M, unsigned N, unsigned K) {
    unsigned int m = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int n = threadIdx.y + blockDim.y * blockIdx.y;
    if(m >= M || n >= N)
        return;
    float c = 0;
    for(unsigned k = 0; k < K; ++k) {
        c += A[m * K + k] * B[k * N + n];
    }
    c = c * alpha;
    float res = c;
    if(beta != 0) {
        res = res + C[m * N + n] * beta;
    }
    C[m * N + n] = res;
}


struct __device_builtin__ __builtin_align__(16) floats4 {
    float data[4];
    __host__ __device__ float operator[](unsigned idx) const { return data[idx]; }

    __host__ __device__ float& operator[](unsigned idx) { return data[idx]; }

    __host__ __device__ floats4 operator*(float other) const {
        return floats4{data[0] * other, data[1] * other, data[2] * other,
                      data[3] * other};
    }

    __host__ __device__ floats4 operator+(const floats4& other) const {
        return floats4{data[0] + other.data[0], data[1] + other.data[1],
                      data[2] + other.data[2], data[3] + other.data[3]};
    }

    // 扩展乘法运算符，用于两个 floats4 之间的乘法
    __host__ __device__ floats4 operator*(const floats4& other) const {
        return floats4{data[0] * other.data[0], data[1] * other.data[1],
                       data[2] * other.data[2], data[3] * other.data[3]};
    }
};

// tile: fetch 4 * floats
__global__ void sgemm_tile_cuda(float* A, float* B, float* C, float alpha, float beta, 
                unsigned M, unsigned N, unsigned K) {
    // 计算当前线程在矩阵 C 中的行和列索引
    unsigned int m = threadIdx.x + blockDim.x * blockIdx.x; // 当前行
    unsigned int n = threadIdx.y + blockDim.y * blockIdx.y; // 当前列

}


int main() {
    unsigned M = 10, K = 10, N = 10;
    float A[M * K];
    float B[K * N];
    float C[M * N];
    float C_tmp[M * N];
    clock_t begin, end;
    float alpha = 0.2, beta = 0.1;
    double GFLOPS;

    sfill_rand_cpu(A, M * K);
    sfill_rand_cpu(B, K * N);
    sfill_rand_cpu(C, M * N);
    scopy_cpu(C_tmp, C, M * N);

    // sprint_2d_cpu(A, M, K);
    // sprint_2d_cpu(B, K, N);
    // sprint_2d_cpu(C, M, N);

    // ---- CPU ----
    begin = clock();
    sgemm_cpu(A, B, C, alpha, beta, M, N, K);
    end = clock();

    scopy_cpu(C, C_tmp, M * N);
    printf("CPU use         : %.6f(ms)\n", double(end - begin) / CLOCKS_PER_SEC * 1e3);

    // ---- CUDA ----
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float milliseconds = 0;
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, M * K * sizeof(float));
    cudaMalloc((void**)&B_d, K * N * sizeof(float));
    cudaMalloc((void**)&C_d, M * N * sizeof(float));
    cudaMemcpy(A_d, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((M - 1)/ block.x + 1, (N - 1) / block.y + 1);
    cudaEventRecord(startEvent);
    sgemm_cuda<<<grid, block>>>(A_d, B_d, C_d, alpha, beta, M, N, K);
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
    cudaMemcpy(C, C_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("GPU use         : %.6f(ms)\n", milliseconds);
    GFLOPS = 2 * 1e-9 * M * N * K / (milliseconds * 1e-3);
    printf("GPU Throughput  : %.6f GFLOPS\n", GFLOPS);


    scopy_cpu(C, C_tmp, M * N);
    cudaMemcpy(C_d, C, M * N * sizeof(float), cudaMemcpyHostToDevice);
    dim3 block_tile(32, 32);
    dim3 grid_tile((M - 1)/ block.x + 1, (N - 1) / block.y + 1);
    cudaEventRecord(startEvent);
    sgemm_tile_cuda<<<grid_tile, block_tile>>>(A_d, B_d, C_d, alpha, beta, M, N, K);
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
    cudaMemcpy(C, C_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("GPU use         : %.6f(ms) (tile)\n", milliseconds);
    GFLOPS = 2 * 1e-9 * M * N * K / (milliseconds * 1e-3);
    printf("GPU Throughput  : %.6f GFLOPS (tile)\n", GFLOPS);


    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaEventDestroy(stopEvent);
    cudaEventDestroy(startEvent);

    return 0;
}