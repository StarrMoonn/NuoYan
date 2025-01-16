#include "mex.h"
#include "matrix.h"
#include <dispatch/dispatch.h>
#include <Accelerate/Accelerate.h>
#include <cstring>
#include <vecLib/vDSP.h>

// 简化版本，直接使用 vsubD 和 vsmulD
inline void compute_difference(const double* input, double* output, int stride, int count, 
                             double scale, bool forward = true) {
    if (forward) {
        vDSP_vsubD(input, 1, input + stride, 1, output, 1, count);
    } else {
        vDSP_vsubD(input + stride, 1, input, 1, output, 1, count);
    }
    vDSP_vsmulD(output, 1, &scale, output, 1, count);
}

// 辅助函数：应用系数并更新内存
inline void apply_coefficients(double* memory, const double* input_values, 
                             const double* b, const double* a, 
                             const double* K, double* output_values,
                             double* temp_values, int count) {
    vDSP_vmulD(memory, 1, b, 1, memory, 1, count);  // memory *= b
    vDSP_vmulD(input_values, 1, a, 1, temp_values, 1, count);  // temp = values * a
    vDSP_vaddD(memory, 1, temp_values, 1, memory, 1, count);  // memory += temp
    
    for(int i = 0; i < count; i++) {
        output_values[i] = input_values[i] / K[i] + memory[i];
    }
}

void compute_wave_propagation_optimized(
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

    // 第一部分：计算正应力场 sigmaxx 和 sigmayy
    dispatch_apply(NY-1, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t j_idx) {
        int j = j_idx + 1;  // j: [1, NY-1]
        
        // 使用 thread_local 避免重复分配
        static thread_local double* temp_dvx_dx = new double[NX];
        static thread_local double* temp_dvy_dy = new double[NX];
        static thread_local double* temp_values = new double[NX];
        
        // 使用辅助函数计算导数
        compute_difference(&vx[j*NX], temp_dvx_dx, 1, NX-1, inv_dx);  // 前向差分
        compute_difference(&vy[j*NX], temp_dvy_dy, NX, NX-1, -inv_dy, false);  // 反向差分
        
        // 使用辅助函数更新内存和计算最终值
        apply_coefficients(&memory_dvx_dx[j*NX], temp_dvx_dx, 
                          b_x_half, a_x_half, K_x_half,
                          temp_dvx_dx, temp_values, NX-1);
        
        apply_coefficients(&memory_dvy_dy[j*NX], temp_dvy_dy,
                          b_y + j, a_y + j, K_y + j,
                          temp_dvy_dy, temp_values, NX-1);
        
        // 批量更新应力分量
        for(int i = 0; i < NX-1; i++) {
            sigmaxx[i + j*NX] += DELTAT * (
                c11[i + j*NX] * temp_dvx_dx[i] + 
                c13[i + j*NX] * temp_dvy_dy[i]
            );
            
            sigmayy[i + j*NX] += DELTAT * (
                c13[i + j*NX] * temp_dvx_dx[i] + 
                c33[i + j*NX] * temp_dvy_dy[i]
            );
        }
    });

    // 第二部分：计算剪应力 sigmaxy
    dispatch_apply(NY-1, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t j) {
        static thread_local double* temp_dvy_dx = new double[NX];
        static thread_local double* temp_dvx_dy = new double[NX];
        static thread_local double* temp_values = new double[NX];
        
        for(int i = 1; i < NX; i++) {
            compute_difference(&vy[(i-1) + j*NX], &temp_dvy_dx[i], 1, 1, inv_dx);
            compute_difference(&vx[i + j*NX], &temp_dvx_dy[i], NX, 1, inv_dy);
        }
        
        apply_coefficients(&memory_dvy_dx[j*NX + 1], temp_dvy_dx + 1,
                          b_x + 1, a_x + 1, K_x + 1,
                          temp_dvy_dx + 1, temp_values, NX-1);
        
        apply_coefficients(&memory_dvx_dy[j*NX + 1], temp_dvx_dy + 1,
                          b_y_half + j, a_y_half + j, K_y_half + j,
                          temp_dvx_dy + 1, temp_values, NX-1);
        
        for(int i = 1; i < NX; i++) {
            sigmaxy[i + j*NX] += c44[i + j*NX] * 
                (temp_dvy_dx[i] + temp_dvx_dy[i]) * DELTAT;
        }
    });

    // 第三部分：计算速度场 vx
    dispatch_apply(NY-2, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t j_idx) {
        int j = j_idx + 1;  // j: [1, NY-1]
        
        static thread_local double* temp_dsigmaxx_dx = new double[NX];
        static thread_local double* temp_dsigmaxy_dy = new double[NX];
        static thread_local double* temp_values = new double[NX];
        
        compute_difference(&sigmaxx[j*NX], temp_dsigmaxx_dx, 1, NX-1, inv_dx);
        compute_difference(&sigmaxy[j*NX], temp_dsigmaxy_dy, NX, NX-1, inv_dy);
        
        apply_coefficients(&memory_dsigmaxx_dx[j*NX + 1], temp_dsigmaxx_dx,
                          b_x + 1, a_x + 1, K_x + 1,
                          temp_dsigmaxx_dx, temp_values, NX-1);
        
        apply_coefficients(&memory_dsigmaxy_dy[j*NX + 1], temp_dsigmaxy_dy,
                          b_y + j, a_y + j, K_y + j,
                          temp_dsigmaxy_dy, temp_values, NX-1);
        
        for(int i = 1; i < NX; i++) {
            vx[i + j*NX] += (temp_dsigmaxx_dx[i-1] + temp_dsigmaxy_dy[i-1]) * 
                           DELTAT / rho[i + j*NX];
        }
    });

    // 第四部分：计算速度场 vy
    dispatch_apply(NY-1, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t j) {
        static thread_local double* temp_dsigmaxy_dx = new double[NX];
        static thread_local double* temp_dsigmayy_dy = new double[NX];
        static thread_local double* temp_values = new double[NX];
        
        compute_difference(&sigmaxy[j*NX], temp_dsigmaxy_dx, 1, NX-1, inv_dx);
        compute_difference(&sigmayy[j*NX], temp_dsigmayy_dy, NX, NX-1, inv_dy);
        
        apply_coefficients(&memory_dsigmaxy_dx[j*NX], temp_dsigmaxy_dx,
                          b_x_half, a_x_half, K_x_half,
                          temp_dsigmaxy_dx, temp_values, NX-1);
        
        apply_coefficients(&memory_dsigmayy_dy[j*NX], temp_dsigmayy_dy,
                          b_y_half + j, a_y_half + j, K_y_half + j,
                          temp_dsigmayy_dy, temp_values, NX-1);
        
        for(int i = 0; i < NX-1; i++) {
            vy[i + j*NX] += (temp_dsigmaxy_dx[i] + temp_dsigmayy_dy[i]) * 
                           DELTAT / rho[i + j*NX];
        }
    });
}

// MEX 入口函数保持不变
extern "C" void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 35) {
        mexErrMsgIdAndTxt("MyToolbox:compute_wave_propagation_optimized:nrhs",
                         "Need 35 input arguments.");
    }
    
    if (nlhs != 2) {
        mexErrMsgIdAndTxt("MyToolbox:compute_wave_propagation_optimized:nlhs",
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
    
    // 继续获取剩余参数
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
    
    double DELTAX = mxGetScalar(prhs[30]);
    double DELTAY = mxGetScalar(prhs[31]);
    double DELTAT = mxGetScalar(prhs[32]);
    int NX = static_cast<int>(mxGetScalar(prhs[33]));
    int NY = static_cast<int>(mxGetScalar(prhs[34]));
    
    // 创建输出数组 - 这里有问题！
    plhs[0] = mxCreateDoubleMatrix(NX, NY, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(NX, NY, mxREAL);
    
    // 这里的问题是：我们创建了新的输出数组，但计算是在输入数组上进行的
    // 需要先复制输入到输出，然后在输出数组上计算
    
    // 获取输出数组指针
    double *out_vx = mxGetPr(plhs[0]);
    double *out_vy = mxGetPr(plhs[1]);
    
    // 先复制输入到输出
    memcpy(out_vx, vx, NX * NY * sizeof(double));
    memcpy(out_vy, vy, NX * NY * sizeof(double));
    
    // 使用输出数组指针调用计算函数
    compute_wave_propagation_optimized(
        out_vx, out_vy,  // 使用输出数组而不是输入数组
        sigmaxx, sigmayy, sigmaxy,
        memory_dvx_dx, memory_dvy_dy, memory_dvy_dx, memory_dvx_dy,
        memory_dsigmaxx_dx, memory_dsigmaxy_dy, memory_dsigmaxy_dx, memory_dsigmayy_dy,
        c11, c13, c33, c44, rho,
        b_x, b_y, b_x_half, b_y_half,
        a_x, a_y, a_x_half, a_y_half,
        K_x, K_y, K_x_half, K_y_half,
        DELTAX, DELTAY, DELTAT, NX, NY);
    
    // 不需要最后的 memcpy，因为计算已经直接在输出数组上完成
} 