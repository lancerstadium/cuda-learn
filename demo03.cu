#include "cuda_runtime.h"
#include <stdio.h>

// ref: https://zhuanlan.zhihu.com/p/34587739

struct Matrix {
    int width;
    int height;
    float *elements;
};

__device__ float getElement(Matrix *A, int row, int col) {
    return A->elements[row * A->width + col];
}

__device__ void setElement(Matrix *A, int row, int col, float val) {
    A->elements[row * A->width + col] = val;
}

__global__ void matMulKernel(Matrix *A, Matrix *B, Matrix *C) {
    float Cval = 0.0f;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = 0; i < A->width; ++i) {
        Cval += getElement(A, row, i) * getElement(B, i, col);
    }
    setElement(C, row, col, Cval);
}

int main()
{
    int width = 1 << 10;
    int height = 1 << 10;
    Matrix *A, *B, *C;
    // 申请托管内存
    cudaMallocManaged((void**)&A, sizeof(Matrix));
    cudaMallocManaged((void**)&B, sizeof(Matrix));
    cudaMallocManaged((void**)&C, sizeof(Matrix));
    int nBytes = width * height * sizeof(float);
    cudaMallocManaged((void**)&A->elements, nBytes);
    cudaMallocManaged((void**)&B->elements, nBytes);
    cudaMallocManaged((void**)&C->elements, nBytes);

    // 初始化数据
    A->height = height;
    A->width = width;
    B->height = height;
    B->width = width;
    C->height = height;
    C->width = width;
    for (int i = 0; i < width * height; ++i)
    {
        A->elements[i] = 1.0;
        B->elements[i] = 2.0;
    }

    // 定义kernel的执行配置
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
        (height + blockSize.y - 1) / blockSize.y);
    // 执行kernel
    matMulKernel << < gridSize, blockSize >> >(A, B, C);


    // 同步device 保证结果能正确访问
    cudaDeviceSynchronize();
    // 检查执行结果
    float maxError = 0.0;
    for (int i = 0; i < width * height; ++i)
        maxError = fmax(maxError, fabs(C->elements[i] - 2 * width));
    printf("最大误差: %.2f\n", maxError);

    return 0;
}