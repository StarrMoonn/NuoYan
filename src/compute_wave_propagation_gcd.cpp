#include "mex.h"
#include "matrix.h"
#include <dispatch/dispatch.h>
#include <cstring>

void compute_wave_propagation_gcd(
    double *vx, double *vy, 
    double *sigmaxx, double *sigmayy, double *sigmaxy,
    double *memory_dvx_dx, double *memory_dvy_dy, 
    double *memory_dvy_dx, double *memory_dvx_dy,
    double *memory_dsigmaxx_dx, double *memory_dsigmaxy_dy, 
    double *memory_dsigmaxy_dx, double *memory_dsigmayy_dy,
    double *c11, double *c13, double *c33, double *c44, double *rho,
    double *b_x, double *b_y, double *b_x_half, double *b_y_half,
    double *a_x, double *a_y, double *a_x_half, double *a_y_half,
    double *K_x, double *K_y, double *K_x_half, double *K_y_half,
    double DELTAX, double DELTAY, double DELTAT, 
    int NX, int NY) {
    
    const double inv_dx = 1.0 / DELTAX;
    const double inv_dy = 1.0 / DELTAY;

    // 创建一个串行队列来确保步骤顺序
    dispatch_queue_t serial_queue = dispatch_queue_create("com.wave.serial", DISPATCH_QUEUE_SERIAL);
    // 创建一个并行队列来执行并行计算
    dispatch_queue_t parallel_queue = dispatch_queue_create("com.wave.parallel", DISPATCH_QUEUE_CONCURRENT);
    
    dispatch_sync(serial_queue, ^{
        // 第一步：计算应力场 sigmaxx 和 sigmayy
        dispatch_apply(NY - 1, parallel_queue, ^(size_t j_idx) {
            int j = j_idx + 1;  // 修正索引，使 j 从 1 到 NY-1
            for (int i = 0; i < NX - 1; i++) {
                double value_dvx_dx = (vx[(i + 1) + j * NX] - vx[i + j * NX]) * inv_dx;
                double value_dvy_dy = (vy[i + j * NX] - vy[i + (j - 1) * NX]) * inv_dy;

                memory_dvx_dx[i + j * NX] = b_x_half[i] * memory_dvx_dx[i + j * NX] + 
                                           a_x_half[i] * value_dvx_dx;
                memory_dvy_dy[i + j * NX] = b_y[j] * memory_dvy_dy[i + j * NX] + 
                                           a_y[j] * value_dvy_dy;

                value_dvx_dx = value_dvx_dx / K_x_half[i] + memory_dvx_dx[i + j * NX];
                value_dvy_dy = value_dvy_dy / K_y[j] + memory_dvy_dy[i + j * NX];

                sigmaxx[i + j * NX] += DELTAT * (
                    c11[i + j * NX] * value_dvx_dx + 
                    c13[i + j * NX] * value_dvy_dy
                );
                
                sigmayy[i + j * NX] += DELTAT * (
                    c13[i + j * NX] * value_dvx_dx + 
                    c33[i + j * NX] * value_dvy_dy
                );
            }
        });
    });

    dispatch_sync(serial_queue, ^{
        // 第二步：计算剪应力 sigmaxy
        dispatch_apply(NY - 1, parallel_queue, ^(size_t j_idx) {
            int j = j_idx;  // j 从 0 到 NY-2
            for (int i = 1; i < NX; i++) {
                double value_dvy_dx = (vy[i + j * NX] - vy[(i - 1) + j * NX]) * inv_dx;
                double value_dvx_dy = (vx[i + (j + 1) * NX] - vx[i + j * NX]) * inv_dy;

                memory_dvy_dx[i + j * NX] = b_x[i] * memory_dvy_dx[i + j * NX] + 
                                           a_x[i] * value_dvy_dx;
                memory_dvx_dy[i + j * NX] = b_y_half[j] * memory_dvx_dy[i + j * NX] + 
                                           a_y_half[j] * value_dvx_dy;

                value_dvy_dx = value_dvy_dx / K_x[i] + memory_dvy_dx[i + j * NX];
                value_dvx_dy = value_dvx_dy / K_y_half[j] + memory_dvx_dy[i + j * NX];

                sigmaxy[i + j * NX] += c44[i + j * NX] * (value_dvy_dx + value_dvx_dy) * DELTAT;
            }
        });
    });

    // 计算 x 方向速度场
    dispatch_sync(serial_queue, ^{
        dispatch_apply(NY - 1, parallel_queue, ^(size_t j_idx) {
            int j = j_idx + 1;  // j 从 1 到 NY-1
            for (int i = 1; i < NX; i++) {
                double value_dsigmaxx_dx = (sigmaxx[i + j * NX] - sigmaxx[(i - 1) + j * NX]) * inv_dx;
                double value_dsigmaxy_dy = (sigmaxy[i + j * NX] - sigmaxy[i + (j - 1) * NX]) * inv_dy;

                memory_dsigmaxx_dx[i + j * NX] = b_x[i] * memory_dsigmaxx_dx[i + j * NX] + 
                                                a_x[i] * value_dsigmaxx_dx;
                memory_dsigmaxy_dy[i + j * NX] = b_y[j] * memory_dsigmaxy_dy[i + j * NX] + 
                                                a_y[j] * value_dsigmaxy_dy;

                value_dsigmaxx_dx = value_dsigmaxx_dx / K_x[i] + memory_dsigmaxx_dx[i + j * NX];
                value_dsigmaxy_dy = value_dsigmaxy_dy / K_y[j] + memory_dsigmaxy_dy[i + j * NX];

                vx[i + j * NX] += (value_dsigmaxx_dx + value_dsigmaxy_dy) * DELTAT / rho[i + j * NX];
            }
        });
    });

    // 计算 y 方向速度场
    dispatch_sync(serial_queue, ^{
        dispatch_apply(NY - 1, parallel_queue, ^(size_t j_idx) {
            int j = j_idx;  // j 从 0 到 NY-2
            for (int i = 0; i < NX - 1; i++) {
                double value_dsigmaxy_dx = (sigmaxy[(i + 1) + j * NX] - sigmaxy[i + j * NX]) * inv_dx;
                double value_dsigmayy_dy = (sigmayy[i + (j + 1) * NX] - sigmayy[i + j * NX]) * inv_dy;

                memory_dsigmaxy_dx[i + j * NX] = b_x_half[i] * memory_dsigmaxy_dx[i + j * NX] + 
                                                a_x_half[i] * value_dsigmaxy_dx;
                memory_dsigmayy_dy[i + j * NX] = b_y_half[j] * memory_dsigmayy_dy[i + j * NX] + 
                                                a_y_half[j] * value_dsigmayy_dy;

                value_dsigmaxy_dx = value_dsigmaxy_dx / K_x_half[i] + memory_dsigmaxy_dx[i + j * NX];
                value_dsigmayy_dy = value_dsigmayy_dy / K_y_half[j] + memory_dsigmayy_dy[i + j * NX];

                vy[i + j * NX] += (value_dsigmaxy_dx + value_dsigmayy_dy) * DELTAT / rho[i + j * NX];
            }
        });
    });

    dispatch_release(serial_queue);
    dispatch_release(parallel_queue);
}

extern "C" void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // 检查输入参数数量
    if (nrhs != 35) {
        mexErrMsgIdAndTxt("MyToolbox:compute_wave_propagation_gcd:nrhs",
                         "Need 35 input arguments.");
    }
    
    // 检查输出参数数量
    if (nlhs != 2) {
        mexErrMsgIdAndTxt("MyToolbox:compute_wave_propagation_gcd:nlhs",
                         "Need 2 output arguments.");
    }
    
    // 获取输入参数
    double *vx = mxGetPr(prhs[0]);
    double *vy = mxGetPr(prhs[1]);
    double *sigmaxx = mxGetPr(prhs[2]);
    double *sigmayy = mxGetPr(prhs[3]);
    double *sigmaxy = mxGetPr(prhs[4]);

    double *memory_dvx_dx = mxGetPr(prhs[5]);
    double *memory_dvy_dy = mxGetPr(prhs[6]);
    double *memory_dvy_dx = mxGetPr(prhs[7]);
    double *memory_dvx_dy = mxGetPr(prhs[8]);
    double *memory_dsigmaxx_dx = mxGetPr(prhs[9]);
    double *memory_dsigmaxy_dy = mxGetPr(prhs[10]);
    double *memory_dsigmaxy_dx = mxGetPr(prhs[11]);
    double *memory_dsigmayy_dy = mxGetPr(prhs[12]);
    
    double *c11 = mxGetPr(prhs[13]);
    double *c13 = mxGetPr(prhs[14]);
    double *c33 = mxGetPr(prhs[15]);
    double *c44 = mxGetPr(prhs[16]);
    double *rho = mxGetPr(prhs[17]);
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

    // 计算参数
    double DELTAX = mxGetScalar(prhs[30]);
    double DELTAY = mxGetScalar(prhs[31]);
    double DELTAT = mxGetScalar(prhs[32]);
    int NX = static_cast<int>(mxGetScalar(prhs[33]));
    int NY = static_cast<int>(mxGetScalar(prhs[34]));
    
    // 调用计算函数
    compute_wave_propagation_gcd(vx, vy, sigmaxx, sigmayy, sigmaxy,
                               memory_dvx_dx, memory_dvy_dy, memory_dvy_dx, memory_dvx_dy,
                               memory_dsigmaxx_dx, memory_dsigmaxy_dy, memory_dsigmaxy_dx, memory_dsigmayy_dy,
                               c11, c13, c33, c44, rho,
                               b_x, b_y, b_x_half, b_y_half,
                               a_x, a_y, a_x_half, a_y_half,
                               K_x, K_y, K_x_half, K_y_half,
                               DELTAX, DELTAY, DELTAT, NX, NY);
    
    // 设置输出矩阵
    plhs[0] = mxCreateDoubleMatrix(NX, NY, mxREAL);
    memcpy(mxGetPr(plhs[0]), vx, NX * NY * sizeof(double));

    plhs[1] = mxCreateDoubleMatrix(NX, NY, mxREAL);
    memcpy(mxGetPr(plhs[1]), vy, NX * NY * sizeof(double));
} 