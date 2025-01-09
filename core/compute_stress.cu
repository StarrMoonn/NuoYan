#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>

// 共同的宏定义
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define IDX(i,j,NY) ((j) + (i)*(NY))  // 列优先存储的索引宏
#define EXPECTED_INPUTS 28
#define EXPECTED_OUTPUTS 7

// 错误检查宏
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            mexErrMsgIdAndTxt("MATLAB:cuda:error", \
                "CUDA error %s", cudaGetErrorString(err)); \
        } \
    } while(0)

__global__ void compute_stress_kernel(
    // 输入数组
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    // 输出数组
    double* __restrict__ sigmaxx,
    double* __restrict__ sigmayy,
    double* __restrict__ sigmaxy,
    // PML记忆变量
    double* __restrict__ memory_dvx_dx,
    double* __restrict__ memory_dvy_dy,
    double* __restrict__ memory_dvy_dx,
    double* __restrict__ memory_dvx_dy,
    // PML参数
    const double* __restrict__ b_x_half,
    const double* __restrict__ b_y,
    const double* __restrict__ a_x_half,
    const double* __restrict__ a_y,
    const double* __restrict__ b_x,
    const double* __restrict__ b_y_half,
    const double* __restrict__ a_x,
    const double* __restrict__ a_y_half,
    const double* __restrict__ K_x_half,
    const double* __restrict__ K_y,
    const double* __restrict__ K_x,
    const double* __restrict__ K_y_half,
    // 常量参数
    const double DELTAX,
    const double DELTAY,
    const double DELTAT,
    const double c11,
    const double c13,
    const double c33,
    const double c44,
    const int NX,
    const int NY)
{
    // 共享内存定义
    __shared__ double s_vx[BLOCK_SIZE_Y+1][BLOCK_SIZE_X+1];
    __shared__ double s_vy[BLOCK_SIZE_Y+1][BLOCK_SIZE_X+1];
    
    // 计算线程索引（1-based）
    const int i = blockIdx.x * BLOCK_SIZE_X + threadIdx.x + 1;
    const int j = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y + 1;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // 加载数据到共享内存
    if (i <= NX && j <= NY) {
        s_vx[ty][tx] = vx[IDX(i,j,NY)];
        s_vy[ty][tx] = vy[IDX(i,j,NY)];
        
        if (tx == BLOCK_SIZE_X-1) {
            s_vx[ty][tx+1] = vx[IDX(i+1,j,NY)];
            s_vy[ty][tx+1] = vy[IDX(i+1,j,NY)];
        }
        if (ty == BLOCK_SIZE_Y-1) {
            s_vx[ty+1][tx] = vx[IDX(i,j+1,NY)];
            s_vy[ty+1][tx] = vy[IDX(i,j+1,NY)];
        }
    }
    
    __syncthreads();
    
    // 计算正应力
    if (j >= 2 && j <= NY && i >= 1 && i <= NX-1) {
        const double value_dvx_dx = (s_vx[ty][tx+1] - s_vx[ty][tx]) / DELTAX;
        const double value_dvy_dy = (s_vy[ty][tx] - s_vy[ty-1][tx]) / DELTAY;
        
        const int idx = IDX(i,j,NY);
        memory_dvx_dx[idx] = b_x_half[i] * memory_dvx_dx[idx] + 
                            a_x_half[i] * value_dvx_dx;
        memory_dvy_dy[idx] = b_y[j] * memory_dvy_dy[idx] + 
                            a_y[j] * value_dvy_dy;
        
        const double dvx_dx_pml = value_dvx_dx / K_x_half[i] + memory_dvx_dx[idx];
        const double dvy_dy_pml = value_dvy_dy / K_y[j] + memory_dvy_dy[idx];
        
        sigmaxx[idx] += (c11 * dvx_dx_pml + c13 * dvy_dy_pml) * DELTAT;
        sigmayy[idx] += (c13 * dvx_dx_pml + c33 * dvy_dy_pml) * DELTAT;
    }
    
    __syncthreads();
    
    // 计算剪切应力
    if (j >= 1 && j <= NY-1 && i >= 2 && i <= NX) {
        const double value_dvy_dx = (s_vy[ty][tx] - s_vy[ty][tx-1]) / DELTAX;
        const double value_dvx_dy = (s_vx[ty+1][tx] - s_vx[ty][tx]) / DELTAY;
        
        const int idx = IDX(i,j,NY);
        memory_dvy_dx[idx] = b_x[i] * memory_dvy_dx[idx] + 
                            a_x[i] * value_dvy_dx;
        memory_dvx_dy[idx] = b_y_half[j] * memory_dvx_dy[idx] + 
                            a_y_half[j] * value_dvx_dy;
        
        const double dvy_dx_pml = value_dvy_dx / K_x[i] + memory_dvy_dx[idx];
        const double dvx_dy_pml = value_dvx_dy / K_y_half[j] + memory_dvx_dy[idx];
        
        sigmaxy[idx] += c44 * (dvy_dx_pml + dvx_dy_pml) * DELTAT;
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // 初始化CUDA
    CHECK_CUDA_ERROR(mxInitGPU());
    
    // 参数检查
    if (nrhs != EXPECTED_INPUTS) {
        mexErrMsgIdAndTxt("MATLAB:compute_stress:invalidNumInputs",
            "Expected %d inputs, got %d.", EXPECTED_INPUTS, nrhs);
    }
    if (nlhs != EXPECTED_OUTPUTS) {
        mexErrMsgIdAndTxt("MATLAB:compute_stress:invalidNumOutputs",
            "Expected %d outputs, got %d.", EXPECTED_OUTPUTS, nlhs);
    }
    
    // 获取输入GPU数组
    mxGPUArray const *vx = mxGPUCreateFromMxArray(prhs[0]);
    mxGPUArray const *vy = mxGPUCreateFromMxArray(prhs[1]);
    mxGPUArray *sigmaxx = mxGPUCopyFromMxArray(prhs[2]);
    mxGPUArray *sigmayy = mxGPUCopyFromMxArray(prhs[3]);
    mxGPUArray *sigmaxy = mxGPUCopyFromMxArray(prhs[4]);
    mxGPUArray *memory_dvx_dx = mxGPUCopyFromMxArray(prhs[5]);
    mxGPUArray *memory_dvy_dy = mxGPUCopyFromMxArray(prhs[6]);
    mxGPUArray *memory_dvy_dx = mxGPUCopyFromMxArray(prhs[7]);
    mxGPUArray *memory_dvx_dy = mxGPUCopyFromMxArray(prhs[8]);
    
    // 获取PML参数数组
    mxGPUArray const *b_x_half = mxGPUCreateFromMxArray(prhs[9]);
    mxGPUArray const *b_y = mxGPUCreateFromMxArray(prhs[10]);
    mxGPUArray const *a_x_half = mxGPUCreateFromMxArray(prhs[11]);
    mxGPUArray const *a_y = mxGPUCreateFromMxArray(prhs[12]);
    mxGPUArray const *b_x = mxGPUCreateFromMxArray(prhs[13]);
    mxGPUArray const *b_y_half = mxGPUCreateFromMxArray(prhs[14]);
    mxGPUArray const *a_x = mxGPUCreateFromMxArray(prhs[15]);
    mxGPUArray const *a_y_half = mxGPUCreateFromMxArray(prhs[16]);
    mxGPUArray const *K_x_half = mxGPUCreateFromMxArray(prhs[17]);
    mxGPUArray const *K_y = mxGPUCreateFromMxArray(prhs[18]);
    mxGPUArray const *K_x = mxGPUCreateFromMxArray(prhs[19]);
    mxGPUArray const *K_y_half = mxGPUCreateFromMxArray(prhs[20]);
    
    // 获取标量参数
    const double DELTAX = mxGetScalar(prhs[21]);
    const double DELTAY = mxGetScalar(prhs[22]);
    const double DELTAT = mxGetScalar(prhs[23]);
    const double c11 = mxGetScalar(prhs[24]);
    const double c13 = mxGetScalar(prhs[25]);
    const double c33 = mxGetScalar(prhs[26]);
    const double c44 = mxGetScalar(prhs[27]);
    
    // 获取数组维度
    mwSize const *dims = mxGPUGetDimensions(vx);
    int NX = (int)dims[0];
    int NY = (int)dims[1];
    
    // 设置CUDA网格和块大小
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 blocksPerGrid(
        (NX + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (NY + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y
    );
    
    // 调用CUDA核函数
    compute_stress_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        (double*)mxGPUGetDataReadOnly(vx),
        (double*)mxGPUGetDataReadOnly(vy),
        (double*)mxGPUGetData(sigmaxx),
        (double*)mxGPUGetData(sigmayy),
        (double*)mxGPUGetData(sigmaxy),
        (double*)mxGPUGetData(memory_dvx_dx),
        (double*)mxGPUGetData(memory_dvy_dy),
        (double*)mxGPUGetData(memory_dvy_dx),
        (double*)mxGPUGetData(memory_dvx_dy),
        (double*)mxGPUGetDataReadOnly(b_x_half),
        (double*)mxGPUGetDataReadOnly(b_y),
        (double*)mxGPUGetDataReadOnly(a_x_half),
        (double*)mxGPUGetDataReadOnly(a_y),
        (double*)mxGPUGetDataReadOnly(b_x),
        (double*)mxGPUGetDataReadOnly(b_y_half),
        (double*)mxGPUGetDataReadOnly(a_x),
        (double*)mxGPUGetDataReadOnly(a_y_half),
        (double*)mxGPUGetDataReadOnly(K_x_half),
        (double*)mxGPUGetDataReadOnly(K_y),
        (double*)mxGPUGetDataReadOnly(K_x),
        (double*)mxGPUGetDataReadOnly(K_y_half),
        DELTAX, DELTAY, DELTAT,
        c11, c13, c33, c44,
        NX, NY
    );
    
    // 检查CUDA错误
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // 创建输出数组
    plhs[0] = mxGPUCreateMxArrayOnGPU(sigmaxx);
    plhs[1] = mxGPUCreateMxArrayOnGPU(sigmayy);
    plhs[2] = mxGPUCreateMxArrayOnGPU(sigmaxy);
    plhs[3] = mxGPUCreateMxArrayOnGPU(memory_dvx_dx);
    plhs[4] = mxGPUCreateMxArrayOnGPU(memory_dvy_dy);
    plhs[5] = mxGPUCreateMxArrayOnGPU(memory_dvy_dx);
    plhs[6] = mxGPUCreateMxArrayOnGPU(memory_dvx_dy);
    
    // 清理GPU数组
    mxGPUDestroyGPUArray(vx);
    mxGPUDestroyGPUArray(vy);
    mxGPUDestroyGPUArray(sigmaxx);
    mxGPUDestroyGPUArray(sigmayy);
    mxGPUDestroyGPUArray(sigmaxy);
    mxGPUDestroyGPUArray(memory_dvx_dx);
    mxGPUDestroyGPUArray(memory_dvy_dy);
    mxGPUDestroyGPUArray(memory_dvy_dx);
    mxGPUDestroyGPUArray(memory_dvx_dy);
    mxGPUDestroyGPUArray(b_x_half);
    mxGPUDestroyGPUArray(b_y);
    mxGPUDestroyGPUArray(a_x_half);
    mxGPUDestroyGPUArray(a_y);
    mxGPUDestroyGPUArray(b_x);
    mxGPUDestroyGPUArray(b_y_half);
    mxGPUDestroyGPUArray(a_x);
    mxGPUDestroyGPUArray(a_y_half);
    mxGPUDestroyGPUArray(K_x_half);
    mxGPUDestroyGPUArray(K_y);
    mxGPUDestroyGPUArray(K_x);
    mxGPUDestroyGPUArray(K_y_half);
} 