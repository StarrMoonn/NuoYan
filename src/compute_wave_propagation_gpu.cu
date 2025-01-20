#include "mex.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <cstring>

// CUDA error checking macro
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            mexPrintf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(err)); \
            mexErrMsgTxt("CUDA error"); \
        } \
    } while (0)

// CUDA kernel: Compute stress field
__global__ void compute_stress_kernel(
    double *vx, double *vy, double *sigmaxx, double *sigmayy, double *sigmaxy,
    double *memory_dvx_dx, double *memory_dvy_dy,
    double *c11, double *c13, double *c33,
    double *b_x_half, double *b_y, double *a_x_half, double *a_y,
    double *K_x_half, double *K_y,
    double DELTAX, double DELTAY, double DELTAT,
    int NX, int NY)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < NX-1 && j < NY) {
        int idx = i + j*NX;
        int idx_ip1 = (i+1) + j*NX;
        int idx_jm1 = i + (j-1)*NX;
        
        double value_dvx_dx = (vx[idx_ip1] - vx[idx]) / DELTAX;
        double value_dvy_dy = (vy[idx] - vy[idx_jm1]) / DELTAY;

        memory_dvx_dx[idx] = b_x_half[i] * memory_dvx_dx[idx] + 
                            a_x_half[i] * value_dvx_dx;
        memory_dvy_dy[idx] = b_y[j] * memory_dvy_dy[idx] + 
                            a_y[j] * value_dvy_dy;

        value_dvx_dx = value_dvx_dx / K_x_half[i] + memory_dvx_dx[idx];
        value_dvy_dy = value_dvy_dy / K_y[j] + memory_dvy_dy[idx];

        sigmaxx[idx] += DELTAT * (
            c11[idx] * value_dvx_dx + 
            c13[idx] * value_dvy_dy
        );
        
        sigmayy[idx] += DELTAT * (
            c13[idx] * value_dvx_dx + 
            c33[idx] * value_dvy_dy
        );
    }
}

// CUDA kernel: Compute shear stress
__global__ void compute_shear_stress_kernel(
    double *vx, double *vy, double *sigmaxy,
    double *memory_dvy_dx, double *memory_dvx_dy,
    double *c44,
    double *b_x, double *b_y_half, double *a_x, double *a_y_half,
    double *K_x, double *K_y_half,
    double DELTAX, double DELTAY, double DELTAT,
    int NX, int NY)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    
    if (i < NX && j < NY-1) {
        int idx = i + j*NX;
        int idx_im1 = (i-1) + j*NX;
        int idx_jp1 = i + (j+1)*NX;

        double value_dvy_dx = (vy[idx] - vy[idx_im1]) / DELTAX;
        double value_dvx_dy = (vx[idx_jp1] - vx[idx]) / DELTAY;

        memory_dvy_dx[idx] = b_x[i] * memory_dvy_dx[idx] + 
                            a_x[i] * value_dvy_dx;
        memory_dvx_dy[idx] = b_y_half[j] * memory_dvx_dy[idx] + 
                            a_y_half[j] * value_dvx_dy;

        value_dvy_dx = value_dvy_dx / K_x[i] + memory_dvy_dx[idx];
        value_dvx_dy = value_dvx_dy / K_y_half[j] + memory_dvx_dy[idx];

        sigmaxy[idx] += c44[idx] * (value_dvy_dx + value_dvx_dy) * DELTAT;
    }
}

// CUDA kernel: Compute x-direction velocity field
__global__ void compute_velocity_x_kernel(
    double *vx, double *sigmaxx, double *sigmaxy,
    double *memory_dsigmaxx_dx, double *memory_dsigmaxy_dy,
    double *rho,
    double *b_x, double *b_y, double *a_x, double *a_y,
    double *K_x, double *K_y,
    double DELTAX, double DELTAY, double DELTAT,
    int NX, int NY)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    
    if (i < NX && j < NY) {
        int idx = i + j*NX;
        int idx_im1 = (i-1) + j*NX;
        int idx_jm1 = i + (j-1)*NX;

        double value_dsigmaxx_dx = (sigmaxx[idx] - sigmaxx[idx_im1]) / DELTAX;
        double value_dsigmaxy_dy = (sigmaxy[idx] - sigmaxy[idx_jm1]) / DELTAY;

        memory_dsigmaxx_dx[idx] = b_x[i] * memory_dsigmaxx_dx[idx] + 
                                 a_x[i] * value_dsigmaxx_dx;
        memory_dsigmaxy_dy[idx] = b_y[j] * memory_dsigmaxy_dy[idx] + 
                                 a_y[j] * value_dsigmaxy_dy;

        value_dsigmaxx_dx = value_dsigmaxx_dx / K_x[i] + memory_dsigmaxx_dx[idx];
        value_dsigmaxy_dy = value_dsigmaxy_dy / K_y[j] + memory_dsigmaxy_dy[idx];

        vx[idx] += (value_dsigmaxx_dx + value_dsigmaxy_dy) * DELTAT / rho[idx];
    }
}

// CUDA kernel: Compute y-direction velocity field
__global__ void compute_velocity_y_kernel(
    double *vy, double *sigmaxy, double *sigmayy,
    double *memory_dsigmaxy_dx, double *memory_dsigmayy_dy,
    double *rho,
    double *b_x_half, double *b_y_half, double *a_x_half, double *a_y_half,
    double *K_x_half, double *K_y_half,
    double DELTAX, double DELTAY, double DELTAT,
    int NX, int NY)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < NX-1 && j < NY-1) {
        int idx = i + j*NX;
        int idx_ip1 = (i+1) + j*NX;
        int idx_jp1 = i + (j+1)*NX;

        double value_dsigmaxy_dx = (sigmaxy[idx_ip1] - sigmaxy[idx]) / DELTAX;
        double value_dsigmayy_dy = (sigmayy[idx_jp1] - sigmayy[idx]) / DELTAY;

        memory_dsigmaxy_dx[idx] = b_x_half[i] * memory_dsigmaxy_dx[idx] + 
                                 a_x_half[i] * value_dsigmaxy_dx;
        memory_dsigmayy_dy[idx] = b_y_half[j] * memory_dsigmayy_dy[idx] + 
                                 a_y_half[j] * value_dsigmayy_dy;

        value_dsigmaxy_dx = value_dsigmaxy_dx / K_x_half[i] + memory_dsigmaxy_dx[idx];
        value_dsigmayy_dy = value_dsigmayy_dy / K_y_half[j] + memory_dsigmayy_dy[idx];

        vy[idx] += (value_dsigmaxy_dx + value_dsigmayy_dy) * DELTAT / rho[idx];
    }
}

// MEX entry function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Check number of input/output parameters
    if (nrhs != 35) {
        mexErrMsgTxt("Need 35 input parameters");
    }
    if (nlhs != 2) {
        mexErrMsgTxt("Need 2 output parameters");
    }

    // Get input array pointers
    double *vx = mxGetPr(prhs[0]);
    double *vy = mxGetPr(prhs[1]);
    double *sigmaxx = mxGetPr(prhs[2]);
    double *sigmayy = mxGetPr(prhs[3]);
    double *sigmaxy = mxGetPr(prhs[4]);
    
    // Get memory variable pointers
    double *memory_dvx_dx = mxGetPr(prhs[5]);
    double *memory_dvy_dy = mxGetPr(prhs[6]);
    double *memory_dvy_dx = mxGetPr(prhs[7]);
    double *memory_dvx_dy = mxGetPr(prhs[8]);
    double *memory_dsigmaxx_dx = mxGetPr(prhs[9]);
    double *memory_dsigmaxy_dy = mxGetPr(prhs[10]);
    double *memory_dsigmaxy_dx = mxGetPr(prhs[11]);
    double *memory_dsigmayy_dy = mxGetPr(prhs[12]);

    // Get material parameter pointers
    double *c11 = mxGetPr(prhs[13]);
    double *c13 = mxGetPr(prhs[14]);
    double *c33 = mxGetPr(prhs[15]);
    double *c44 = mxGetPr(prhs[16]);
    double *rho = mxGetPr(prhs[17]);

    // Get PML parameter pointers
    double *b_x = mxGetPr(prhs[18]);
    double *b_y = mxGetPr(prhs[19]);
    double *b_x_half = mxGetPr(prhs[20]);
    double *b_y_half = mxGetPr(prhs[21]);
    double *a_x = mxGetPr(prhs[22]);
    double *a_y = mxGetPr(prhs[23]);
    double *a_x_half = mxGetPr(prhs[24]);
    double *a_y_half = mxGetPr(prhs[25]);
    double *K_x = mxGetPr(prhs[26]);
    double *K_y = mxGetPr(prhs[27]);
    double *K_x_half = mxGetPr(prhs[28]);
    double *K_y_half = mxGetPr(prhs[29]);   

    // Get computation parameters
    double DELTAX = mxGetScalar(prhs[30]);
    double DELTAY = mxGetScalar(prhs[31]);
    double DELTAT = mxGetScalar(prhs[32]);
    int NX = static_cast<int>(mxGetScalar(prhs[33]));
    int NY = static_cast<int>(mxGetScalar(prhs[34]));

    // Allocate GPU memory - 2D arrays
    size_t size_2d = NX * NY * sizeof(double);
    double *d_vx, *d_vy, *d_sigmaxx, *d_sigmayy, *d_sigmaxy;
    double *d_memory_dvx_dx, *d_memory_dvy_dy, *d_memory_dvy_dx, *d_memory_dvx_dy;
    double *d_memory_dsigmaxx_dx, *d_memory_dsigmaxy_dy;
    double *d_memory_dsigmaxy_dx, *d_memory_dsigmayy_dy;
    double *d_c11, *d_c13, *d_c33, *d_c44, *d_rho;
    double *d_b_x, *d_b_y, *d_b_x_half, *d_b_y_half;
    double *d_a_x, *d_a_y, *d_a_x_half, *d_a_y_half;
    double *d_K_x, *d_K_y, *d_K_x_half, *d_K_y_half;

    // Allocate field variables
    CHECK_CUDA_ERROR(cudaMalloc(&d_vx, size_2d));
    CHECK_CUDA_ERROR(cudaMalloc(&d_vy, size_2d));
    CHECK_CUDA_ERROR(cudaMalloc(&d_sigmaxx, size_2d));
    CHECK_CUDA_ERROR(cudaMalloc(&d_sigmayy, size_2d));
    CHECK_CUDA_ERROR(cudaMalloc(&d_sigmaxy, size_2d));

    // Allocate memory variables
    CHECK_CUDA_ERROR(cudaMalloc(&d_memory_dvx_dx, size_2d));
    CHECK_CUDA_ERROR(cudaMalloc(&d_memory_dvy_dy, size_2d));
    CHECK_CUDA_ERROR(cudaMalloc(&d_memory_dvy_dx, size_2d));
    CHECK_CUDA_ERROR(cudaMalloc(&d_memory_dvx_dy, size_2d));
    CHECK_CUDA_ERROR(cudaMalloc(&d_memory_dsigmaxx_dx, size_2d));
    CHECK_CUDA_ERROR(cudaMalloc(&d_memory_dsigmaxy_dy, size_2d));
    CHECK_CUDA_ERROR(cudaMalloc(&d_memory_dsigmaxy_dx, size_2d));
    CHECK_CUDA_ERROR(cudaMalloc(&d_memory_dsigmayy_dy, size_2d));

    // Allocate material parameters
    CHECK_CUDA_ERROR(cudaMalloc(&d_c11, size_2d));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c13, size_2d));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c33, size_2d));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c44, size_2d));
    CHECK_CUDA_ERROR(cudaMalloc(&d_rho, size_2d));

    // Allocate PML parameters
    size_t size_x = NX * sizeof(double);
    size_t size_y = NY * sizeof(double);

    //  (NX,1)
    CHECK_CUDA_ERROR(cudaMalloc(&d_b_x, size_x));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b_x_half, size_x));
    CHECK_CUDA_ERROR(cudaMalloc(&d_a_x, size_x));
    CHECK_CUDA_ERROR(cudaMalloc(&d_a_x_half, size_x));
    CHECK_CUDA_ERROR(cudaMalloc(&d_K_x, size_x));
    CHECK_CUDA_ERROR(cudaMalloc(&d_K_x_half, size_x));

    //  (NY,1)
    CHECK_CUDA_ERROR(cudaMalloc(&d_b_y, size_y));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b_y_half, size_y));
    CHECK_CUDA_ERROR(cudaMalloc(&d_a_y, size_y));
    CHECK_CUDA_ERROR(cudaMalloc(&d_a_y_half, size_y));
    CHECK_CUDA_ERROR(cudaMalloc(&d_K_y, size_y));
    CHECK_CUDA_ERROR(cudaMalloc(&d_K_y_half, size_y));

    // Copy data to GPU - 2D arrays
    CHECK_CUDA_ERROR(cudaMemcpy(d_vx, vx, size_2d, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_vy, vy, size_2d, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sigmaxx, sigmaxx, size_2d, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sigmayy, sigmayy, size_2d, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sigmaxy, sigmaxy, size_2d, cudaMemcpyHostToDevice));

    // Copy memory variables
    CHECK_CUDA_ERROR(cudaMemcpy(d_memory_dvx_dx, memory_dvx_dx, size_2d, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_memory_dvy_dy, memory_dvy_dy, size_2d, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_memory_dvy_dx, memory_dvy_dx, size_2d, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_memory_dvx_dy, memory_dvx_dy, size_2d, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_memory_dsigmaxx_dx, memory_dsigmaxx_dx, size_2d, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_memory_dsigmaxy_dy, memory_dsigmaxy_dy, size_2d, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_memory_dsigmaxy_dx, memory_dsigmaxy_dx, size_2d, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_memory_dsigmayy_dy, memory_dsigmayy_dy, size_2d, cudaMemcpyHostToDevice));

    // Copy material parameters
    CHECK_CUDA_ERROR(cudaMemcpy(d_c11, c11, size_2d, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_c13, c13, size_2d, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_c33, c33, size_2d, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_c44, c44, size_2d, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_rho, rho, size_2d, cudaMemcpyHostToDevice));

    // Copy PML parameters
    CHECK_CUDA_ERROR(cudaMemcpy(d_b_x, b_x, size_x, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b_y, b_y, size_y, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b_x_half, b_x_half, size_x, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b_y_half, b_y_half, size_y, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_a_x, a_x, size_x, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_a_y, a_y, size_y, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_a_x_half, a_x_half, size_x, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_a_y_half, a_y_half, size_y, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_K_x, K_x, size_x, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_K_y, K_y, size_y, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_K_x_half, K_x_half, size_x, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_K_y_half, K_y_half, size_y, cudaMemcpyHostToDevice));

    // Set thread block and grid dimensions
    dim3 blockSize(16,16);
    dim3 gridSize(
        (NX + blockSize.x - 1) / blockSize.x,
        (NY + blockSize.y - 1) / blockSize.y
    );
    
    // Launch kernels
    compute_stress_kernel<<<gridSize, blockSize>>>(d_vx, d_vy, d_sigmaxx, d_sigmayy, d_sigmaxy,
                                                d_memory_dvx_dx, d_memory_dvy_dy,
                                                d_c11, d_c13, d_c33,
                                                d_b_x_half, d_b_y, d_a_x_half, d_a_y,
                                                d_K_x_half, d_K_y,
                                                DELTAX, DELTAY, DELTAT, NX, NY);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());  

    compute_shear_stress_kernel<<<gridSize, blockSize>>>(d_vx, d_vy, d_sigmaxy,
                                                        d_memory_dvy_dx, d_memory_dvx_dy,
                                                        d_c44,
                                                        d_b_x, d_b_y_half, d_a_x, d_a_y_half,
                                                        d_K_x, d_K_y_half,
                                                        DELTAX, DELTAY, DELTAT, NX, NY);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());  

    compute_velocity_x_kernel<<<gridSize, blockSize>>>(d_vx, d_sigmaxx, d_sigmaxy,
                                                      d_memory_dsigmaxx_dx, d_memory_dsigmaxy_dy,
                                                      d_rho,
                                                      d_b_x, d_b_y, d_a_x, d_a_y,
                                                      d_K_x, d_K_y,
                                                      DELTAX, DELTAY, DELTAT, NX, NY);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());  

    compute_velocity_y_kernel<<<gridSize, blockSize>>>(d_vy, d_sigmaxy, d_sigmayy,
                                                      d_memory_dsigmaxy_dx, d_memory_dsigmayy_dy,
                                                      d_rho,
                                                      d_b_x_half, d_b_y_half, d_a_x_half, d_a_y_half,
                                                      d_K_x_half, d_K_y_half,
                                                      DELTAX, DELTAY, DELTAT, NX, NY);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());  

    // Check for kernel execution errors
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(vx, d_vx, size_2d, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(vy, d_vy, size_2d, cudaMemcpyDeviceToHost));

    // Create output matrices
    plhs[0] = mxCreateDoubleMatrix(NX, NY, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(NX, NY, mxREAL);
    
    // Copy results to output matrices
    memcpy(mxGetPr(plhs[0]), vx, size_2d);
    memcpy(mxGetPr(plhs[1]), vy, size_2d);

    // Free GPU memory
    cudaFree(d_vx); cudaFree(d_vy);
    cudaFree(d_sigmaxx); cudaFree(d_sigmayy); cudaFree(d_sigmaxy);
    cudaFree(d_memory_dvx_dx); cudaFree(d_memory_dvy_dy);
    cudaFree(d_memory_dvy_dx); cudaFree(d_memory_dvx_dy);
    cudaFree(d_memory_dsigmaxx_dx); cudaFree(d_memory_dsigmaxy_dy);
    cudaFree(d_memory_dsigmaxy_dx); cudaFree(d_memory_dsigmayy_dy);
    cudaFree(d_c11); cudaFree(d_c13); cudaFree(d_c33); cudaFree(d_c44);
    cudaFree(d_rho);
    cudaFree(d_b_x); cudaFree(d_b_y);
    cudaFree(d_b_x_half); cudaFree(d_b_y_half);
    cudaFree(d_a_x); cudaFree(d_a_y);
    cudaFree(d_a_x_half); cudaFree(d_a_y_half);
    cudaFree(d_K_x); cudaFree(d_K_y);
    cudaFree(d_K_x_half); cudaFree(d_K_y_half);
} 