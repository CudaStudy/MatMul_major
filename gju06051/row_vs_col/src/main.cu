#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define M_SIZE (512)
#define K_SIZE (512)
#define N_SIZE (512)

#define BLOCK_SIZE (16)

#define CPU_OP 0

#define GPU_1_OP 1
#define GPU_1_T_H2D 2
#define GPU_1_T_D2H 3

#define GPU_2_OP 4
#define GPU_2_T_H2D 5
#define GPU_2_T_D2H 6

#define INDEX2ROW(_index, _width) (int)((_index) / (_width))
#define INDEX2COL(_index, _width) (int)((_index) % (_width))
#define ID2INDEX(_row, _col, _width) (int)(((_row) * (_width)) + (_col))

__global__ void MATMUL_GPU_1(float *_matA, float *_matB, float *_matC, int _m, int _n, int _k)
{
    // thread row, col value get
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    // prevent out range operation
    if (row >= _m || col >= _n)
    {
        return;
    }

    // matrix multiplication
    float val = 0;
    for (int i = 0; i < _k; i++)
    {
        val += __fmul_rn(_matA[ID2INDEX(row, i, _k)], _matB[ID2INDEX(i, col, _n)]);
    }
    // assign
    _matC[ID2INDEX(row, col, _n)] = val;
    return;
}

__global__ void MATMUL_GPU_2(float *_matA, float *_matB, float *_matC, int _m, int _n, int _k)
{
    // thread row, col value get
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    // prevent out range operation
    if (row >= _m || col >= _n)
    {
        return;
    }

    // matrix multiplication
    float val = 0;
    for (int i = 0; i < _k; i++)
    {
        val += __fmul_rn(_matA[ID2INDEX(row, i, _k)], _matB[ID2INDEX(i, col, _n)]);
    }
    // assign
    _matC[ID2INDEX(row, col, _n)] = val;
    return;
}

int main(int argc, char *argv[])
{
    printf("====================Application Start=====================\n");

    // timer setting
    DS_timer timer(7);
    timer.setTimerName(CPU_OP, (char *)"[CPU1] : OP");

    timer.setTimerName(GPU_1_OP, (char *)"[GPU1] : OP");
    timer.setTimerName(GPU_1_T_H2D, (char *)"[GPU1] : Host->Device");
    timer.setTimerName(GPU_1_T_D2H, (char *)"[GPU1] : Device->Host");

    timer.setTimerName(GPU_2_OP, (char *)"[GPU2] : OP");
    timer.setTimerName(GPU_2_T_H2D, (char *)"[GPU2] : Host->Device");
    timer.setTimerName(GPU_2_T_D2H, (char *)"[GPU2] : Device->Host");

    // argument of matrix info
    int m, n, k;
    if (argc < 3)
    {
        // default
        m = M_SIZE;
        k = K_SIZE;
        n = N_SIZE;
    }
    else
    {
        // user argument
        m = atoi(argv[1]);
        k = atoi(argv[2]);
        n = atoi(argv[3]);
    }
    printf("Matrix Size : A : (%d, %d), B : (%d, %d), C : (%d, %d)\n", m, k, k, n, m, n);

    // matrix size for allocation
    long unsigned mat_a_size = m * k;
    long unsigned mat_b_size = k * n;
    long unsigned mat_c_size = m * n;

    // host memory allocation
    float *H_A = new float[mat_a_size];
    float *H_B = new float[mat_b_size];
    float *H_C = new float[mat_c_size];
    float *G1_C = new float[mat_c_size];
    float *G2_C = new float[mat_c_size];

    memset(H_A, 0, sizeof(float) * mat_a_size);
    memset(H_B, 0, sizeof(float) * mat_b_size);
    memset(H_C, 1, sizeof(float) * mat_c_size);
    memset(G1_C, 1, sizeof(float) * mat_c_size);
    memset(G2_C, 1, sizeof(float) * mat_c_size);

    printf("CPU memory allocation\n");

    // generate input matrices
    for (int i = 0; i < mat_a_size; i++)
    {
        H_A[i] = ((rand() % 10) + ((rand() % 100) / 100.0));
    }
    for (int i = 0; i < mat_b_size; i++)
    {
        H_B[i] = ((rand() % 10) + ((rand() % 100) / 100.0));
    }

    printf("CPU matrix randomaization\n");

    // CPU Matrix Multiplication
    timer.onTimer(CPU_OP);
    for (int row = 0; row < m; row++)
    {
        for (int col = 0; col < n; col++)
        {
            int idx = ID2INDEX(row, col, n);
            H_C[idx] = 0;
            for (int i = 0; i < k; i++)
            {
                H_C[idx] += H_A[ID2INDEX(row, i, k)] * H_B[ID2INDEX(i, col, n)];
            }
        }
    }
    timer.offTimer(CPU_OP);

    printf("CPU matrix multilplication\n");

    // device memory allocation
    float *D1_A, *D1_B, *D1_C;
    D1_A = D1_B = D1_C = NULL;

    float *D2_A, *D2_B, *D2_C;
    D2_A = D2_B = D2_C = NULL;

    cudaMalloc(&D1_A, sizeof(float) * mat_a_size);
    cudaMalloc(&D1_B, sizeof(float) * mat_b_size);
    cudaMalloc(&D1_C, sizeof(float) * mat_c_size);

    cudaMalloc(&D2_A, sizeof(float) * mat_a_size);
    cudaMalloc(&D2_B, sizeof(float) * mat_b_size);
    cudaMalloc(&D2_C, sizeof(float) * mat_c_size);

    printf("GPU memory allocation\n");

    // operand trans : Host -> Device
    timer.onTimer(GPU_1_T_H2D);
    cudaMemcpy(D1_A, H_A, sizeof(float) * mat_a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(D1_B, H_B, sizeof(float) * mat_b_size, cudaMemcpyHostToDevice);
    timer.offTimer(GPU_1_T_H2D);

    timer.onTimer(GPU_2_T_H2D);
    cudaMemcpy(D2_A, H_A, sizeof(float) * mat_a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(D2_B, H_B, sizeof(float) * mat_b_size, cudaMemcpyHostToDevice);
    timer.offTimer(GPU_2_T_H2D);

    printf("Operand transition : Host -> Device\n");

    dim3 gridDim(ceil((float)m / BLOCK_SIZE), ceil((float)n / BLOCK_SIZE));
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    printf("Grid, Block : Grid : (%d, %d), Block : (%d, %d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // GPU matrix multiplication
    timer.onTimer(GPU_1_OP);
    MATMUL_GPU_1<<<gridDim, blockDim>>>(D1_A, D1_B, D1_C, m, n, k);
    cudaDeviceSynchronize();
    timer.offTimer(GPU_1_OP);

    timer.onTimer(GPU_2_OP);
    MATMUL_GPU_2<<<gridDim, blockDim>>>(D2_A, D2_B, D2_C, m, n, k);
    cudaDeviceSynchronize();
    timer.offTimer(GPU_2_OP);

    printf("GPU matrix multiplication\n");

    // operand trans : Device -> Host
    timer.onTimer(GPU_1_T_D2H);
    cudaMemcpy(G1_C, D1_C, sizeof(float) * mat_c_size, cudaMemcpyDeviceToHost);
    timer.offTimer(GPU_1_T_D2H);

    timer.onTimer(GPU_2_T_D2H);
    cudaMemcpy(G2_C, D2_C, sizeof(float) * mat_c_size, cudaMemcpyDeviceToHost);
    timer.offTimer(GPU_2_T_D2H);

    printf("Operand transition : Device -> Host\n");

    // operation check
    bool result_1 = true;
    for (int i = 0; i < mat_c_size; i++)
    {
        if (H_C[i] != G1_C[i])
        {
            printf("[%d] %.2f, %.2f\n", i, H_C[i], G1_C[i]);
            result_1 = false;
        }
    }
    if (result_1)
    {
        printf("GPU1 is True\n");
    }
    else
    {
        printf("GPU1 is false\n");
    }

    bool result_2 = true;
    for (int i = 0; i < mat_c_size; i++)
    {
        if (H_C[i] != G2_C[i])
        {
            printf("[%d] %.2f, %.2f\n", i, H_C[i], G2_C[i]);
            result_2 = false;
        }
    }
    if (result_2)
    {
        printf("GPU2 is True\n");
    }
    else
    {
        printf("GPU2 is false\n");
    }

    printf("====================Application End====================\n");
    timer.printTimer();
    printf("\n");
}
