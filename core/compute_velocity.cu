#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>

// 共同的宏定义
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define IDX(i,j,NY) ((j) + (i)*(NY))  // 列优先存储的索引宏
#define EXPECTED_INPUTS 25
#define EXPECTED_OUTPUTS 6

// 错误检查宏
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            mexErrMsgIdAndTxt("MATLAB:cuda:error", \
                "CUDA error %s", cudaGetErrorString(err)); \
        } \
    } while(0)

__global__ void compute_velocity_kernel(
    // 输入数组
    const double* __restrict__ sigmaxx,
    const double* __restrict__ sigmayy,
    const double* __restrict__ sigmaxy,
    // 输出数组
    double* __restrict__ vx,
    double* __restrict__ vy,
    // PML记忆变量
    double* __restrict__ memory_dsigmaxx_dx,
    double* __restrict__ memory_dsigmaxy_dy,
    double* __restrict__ memory_dsigmaxy_dx,
    double* __restrict__ memory_dsigmayy_dy,
    // PML参数
    const double* __restrict__ b_x,
    const double* __restrict__ b_y,
    const double* __restrict__ a_x,
    const double* __restrict__ a_y,
    const double* __restrict__ b_x_half,
    const double* __restrict__ b_y_half,
    const double* __restrict__ a_x_half,
    const double* __restrict__ a_y_half,
    const double* __restrict__ K_x,
    const double* __restrict__ K_y,
    const double* __restrict__ K_x_half,
    const double* __restrict__ K_y_half,
    // 常量参数
    const double DELTAX,
    const double DELTAY,
    const double DELTAT,
    const double rho,
    const int NX,
    const int NY)
{
    // 共享内存定义
    __shared__ double s_sigmaxx[BLOCK_SIZE_Y+1][BLOCK_SIZE_X+1];
    __shared__ double s_sigmayy[BLOCK_SIZE_Y+1][BLOCK_SIZE_X+1];
    __shared__ double s_sigmaxy[BLOCK_SIZE_Y+1][BLOCK_SIZE_X+1];
    
    // 计算线程索引（1-based）
    const int i = blockIdx.x * BLOCK_SIZE_X + threadIdx.x + 1;
    const int j = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y + 1;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // 加载数据到共享内存
    if (i <= NX && j <= NY) {
        s_sigmaxx[ty][tx] = sigmaxx[IDX(i,j,NY)];
        s_sigmayy[ty][tx] = sigmayy[IDX(i,j,NY)];
        s_sigmaxy[ty][tx] = sigmaxy[IDX(i,j,NY)];
        
        if (tx == BLOCK_SIZE_X-1) {
            s_sigmaxx[ty][tx+1] = sigmaxx[IDX(i+1,j,NY)];
            s_sigmayy[ty][tx+1] = sigmayy[IDX(i+1,j,NY)];
            s_sigmaxy[ty][tx+1] = sigmaxy[IDX(i+1,j,NY)];
        }
        if (ty == BLOCK_SIZE_Y-1) {
            s_sigmaxx[ty+1][tx] = sigmaxx[IDX(i,j+1,NY)];
            s_sigmayy[ty+1][tx] = sigmayy[IDX(i,j+1,NY)];
            s_sigmaxy[ty+1][tx] = sigmaxy[IDX(i,j+1,NY)];
        }
    }
    
    __syncthreads();
    
    // 计算x方向速度
    if (j >= 2 && j <= NY && i >= 2 && i <= NX) {
        const double value_dsigmaxx_dx = (s_sigmaxx[ty][tx] - s_sigmaxx[ty][tx-1]) / DELTAX;
        const double value_dsigmaxy_dy = (s_sigmaxy[ty][tx] - s_sigmaxy[ty-1][tx]) / DELTAY;
        
        const int idx = IDX(i,j,NY);
        memory_dsigmaxx_dx[idx] = b_x[i] * memory_dsigmaxx_dx[idx] + 
                                 a_x[i] * value_dsigmaxx_dx;
        memory_dsigmaxy_dy[idx] = b_y[j] * memory_dsigmaxy_dy[idx] + 
                                 a_y[j] * value_dsigmaxy_dy;
        
        const double dsigmaxx_dx_pml = value_dsigmaxx_dx / K_x[i] + memory_dsigmaxx_dx[idx];
        const double dsigmaxy_dy_pml = value_dsigmaxy_dy / K_y[j] + memory_dsigmaxy_dy[idx];
        
        vx[idx] += (dsigmaxx_dx_pml + dsigmaxy_dy_pml) * DELTAT / rho;
    }
    
    __syncthreads();
    
    // 计算y方向速度
    if (j >= 1 && j <= NY-1 && i >= 1 && i <= NX-1) {
        const double value_dsigmaxy_dx = (s_sigmaxy[ty][tx+1] - s_sigmaxy[ty][tx]) / DELTAX;
        const double value_dsigmayy_dy = (s_sigmayy[ty+1][tx] - s_sigmayy[ty][tx]) / DELTAY;
        
        const int idx = IDX(i,j,NY);
        memory_dsigmaxy_dx[idx] = b_x_half[i] * memory_dsigmaxy_dx[idx] + 
                                 a_x_half[i] * value_dsigmaxy_dx;
        memory_dsigmayy_dy[idx] = b_y_half[j] * memory_dsigmayy_dy[idx] + 
                                 a_y_half[j] * value_dsigmayy_dy;
        
        const double dsigmaxy_dx_pml = value_dsigmaxy_dx / K_x_half[i] + memory_dsigmaxy_dx[idx];
        const double dsigmayy_dy_pml = value_dsigmayy_dy / K_y_half[j] + memory_dsigmayy_dy[idx];
        
        vy[idx] += (dsigmaxy_dx_pml + dsigmayy_dy_pml) * DELTAT / rho;
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // 初始化CUDA
    CHECK_CUDA_ERROR(mxInitGPU());
    
    // 参数检查
    if (nrhs != EXPECTED_INPUTS) {
        mexErrMsgIdAndTxt("MATLAB:compute_velocity:invalidNumInputs",
            "Expected %d inputs, got %d.", EXPECTED_INPUTS, nrhs);
    }
    if (nlhs != EXPECTED_OUTPUTS) {
        mexErrMsgIdAndTxt("MATLAB:compute_velocity:invalidNumOutputs",
            "Expected %d outputs, got %d.", EXPECTED_OUTPUTS, nlhs);
    }
    
    // 获取输入GPU数组
    mxGPUArray const *sigmaxx = mxGPUCreateFromMxArray(prhs[0]);
    mxGPUArray const *sigmayy = mxGPUCreateFromMxArray(prhs[1]);
    mxGPUArray const *sigmaxy = mxGPUCreateFromMxArray(prhs[2]);
    mxGPUArray *vx = mxGPUCopyFromMxArray(prhs[3]);
    mxGPUArray *vy = mxGPUCopyFromMxArray(prhs[4]);
    mxGPUArray *memory_dsigmaxx_dx = mxGPUCopyFromMxArray(prhs[5]);
    mxGPUArray *memory_dsigmaxy_dy = mxGPUCopyFromMxArray(prhs[6]);
    mxGPUArray *memory_dsigmaxy_dx = mxGPUCopyFromMxArray(prhs[7]);
    mxGPUArray *memory_dsigmayy_dy = mxGPUCopyFromMxArray(prhs[8]);
    
    // 获取PML参数数组
    mxGPUArray const *b_x = mxGPUCreateFromMxArray(prhs[9]);
    mxGPUArray const *b_y = mxGPUCreateFromMxArray(prhs[10]);
    mxGPUArray const *a_x = mxGPUCreateFromMxArray(prhs[11]);
    mxGPUArray const *a_y = mxGPUCreateFromMxArray(prhs[12]);
    mxGPUArray const *b_x_half = mxGPUCreateFromMxArray(prhs[13]);
    mxGPUArray const *b_y_half = mxGPUCreateFromMxArray(prhs[14]);
    mxGPUArray const *a_x_half = mxGPUCreateFromMxArray(prhs[15]);
    mxGPUArray const *a_y_half = mxGPUCreateFromMxArray(prhs[16]);
    mxGPUArray const *K_x = mxGPUCreateFromMxArray(prhs[17]);
    mxGPUArray const *K_y = mxGPUCreateFromMxArray(prhs[18]);
    mxGPUArray const *K_x_half = mxGPUCreateFromMxArray(prhs[19]);
    mxGPUArray const *K_y_half = mxGPUCreateFromMxArray(prhs[20]);
    
    // 获取标量参数
    const double DELTAX = mxGetScalar(prhs[21]);
    const double DELTAY = mxGetScalar(prhs[22]);
    const double DELTAT = mxGetScalar(prhs[23]);
    const double rho = mxGetScalar(prhs[24]);
    
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
    compute_velocity_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        (double*)mxGPUGetDataReadOnly(sigmaxx),
        (double*)mxGPUGetDataReadOnly(sigmayy),
        (double*)mxGPUGetDataReadOnly(sigmaxy),
        (double*)mxGPUGetData(vx),
        (double*)mxGPUGetData(vy),
        (double*)mxGPUGetData(memory_dsigmaxx_dx),
        (double*)mxGPUGetData(memory_dsigmaxy_dy),
        (double*)mxGPUGetData(memory_dsigmaxy_dx),
        (double*)mxGPUGetData(memory_dsigmayy_dy),
        (double*)mxGPUGetDataReadOnly(b_x),
        (double*)mxGPUGetDataReadOnly(b_y),
        (double*)mxGPUGetDataReadOnly(a_x),
        (double*)mxGPUGetDataReadOnly(a_y),
        (double*)mxGPUGetDataReadOnly(b_x_half),
        (double*)mxGPUGetDataReadOnly(b_y_half),
        (double*)mxGPUGetDataReadOnly(a_x_half),
        (double*)mxGPUGetDataReadOnly(a_y_half),
        (double*)mxGPUGetDataReadOnly(K_x),
        (double*)mxGPUGetDataReadOnly(K_y),
        (double*)mxGPUGetDataReadOnly(K_x_half),
        (double*)mxGPUGetDataReadOnly(K_y_half),
        DELTAX, DELTAY, DELTAT, rho,
        NX, NY
    );
    
    // 检查CUDA错误
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // 创建输出数组
    plhs[0] = mxGPUCreateMxArrayOnGPU(vx);
    plhs[1] = mxGPUCreateMxArrayOnGPU(vy);
    plhs[2] = mxGPUCreateMxArrayOnGPU(memory_dsigmaxx_dx);
    plhs[3] = mxGPUCreateMxArrayOnGPU(memory_dsigmaxy_dy);
    plhs[4] = mxGPUCreateMxArrayOnGPU(memory_dsigmaxy_dx);
    plhs[5] = mxGPUCreateMxArrayOnGPU(memory_dsigmayy_dy);
    
    // 清理GPU数组
    mxGPUDestroyGPUArray(sigmaxx);
    mxGPUDestroyGPUArray(sigmayy);
    mxGPUDestroyGPUArray(sigmaxy);
    mxGPUDestroyGPUArray(vx);
    mxGPUDestroyGPUArray(vy);
    mxGPUDestroyGPUArray(memory_dsigmaxx_dx);
    mxGPUDestroyGPUArray(memory_dsigmaxy_dy);
    mxGPUDestroyGPUArray(memory_dsigmaxy_dx);
    mxGPUDestroyGPUArray(memory_dsigmayy_dy);
    mxGPUDestroyGPUArray(b_x);
    mxGPUDestroyGPUArray(b_y);
    mxGPUDestroyGPUArray(a_x);
    mxGPUDestroyGPUArray(a_y);
    mxGPUDestroyGPUArray(b_x_half);
    mxGPUDestroyGPUArray(b_y_half);
    mxGPUDestroyGPUArray(a_x_half);
    mxGPUDestroyGPUArray(a_y_half);
    mxGPUDestroyGPUArray(K_x);
    mxGPUDestroyGPUArray(K_y);
    mxGPUDestroyGPUArray(K_x_half);
    mxGPUDestroyGPUArray(K_y_half);
} 