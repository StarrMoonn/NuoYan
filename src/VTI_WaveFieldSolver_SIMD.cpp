#include "mex.h"
#include "matrix.h"
#include <omp.h>
#include <cstring>
#include <immintrin.h>

// 定义内存对齐大小
#define ALIGN_SIZE 32  // AVX2需要32字节对齐
#define SIMD_WIDTH 4   // 每次处理4个double

void compute_wave_propagation_simd(
    double* __restrict vx, double* __restrict vy,
    double* __restrict sigmaxx, double* __restrict sigmayy, 
    double* __restrict sigmaxy,
    double* __restrict memory_dvx_dx, double* __restrict memory_dvy_dy,
    double* __restrict memory_dvy_dx, double* __restrict memory_dvx_dy,
    double* __restrict memory_dsigmaxx_dx, double* __restrict memory_dsigmaxy_dy,
    double* __restrict memory_dsigmaxy_dx, double* __restrict memory_dsigmayy_dy,
    const double* __restrict c11, const double* __restrict c13,
    const double* __restrict c33, const double* __restrict c44,
    const double* __restrict rho,
    const double* __restrict b_x, const double* __restrict b_y,
    const double* __restrict b_x_half, const double* __restrict b_y_half,
    const double* __restrict a_x, const double* __restrict a_y,
    const double* __restrict a_x_half, const double* __restrict a_y_half,
    const double* __restrict K_x, const double* __restrict K_y,
    const double* __restrict K_x_half, const double* __restrict K_y_half,
    const double DELTAX, const double DELTAY, const double DELTAT,
    const int NX, const int NY) {

    const double inv_dx = 1.0 / DELTAX;
    const double inv_dy = 1.0 / DELTAY;
    const __m256d inv_dx_vec = _mm256_set1_pd(inv_dx);
    const __m256d inv_dy_vec = _mm256_set1_pd(inv_dy);
    const __m256d dt_vec = _mm256_set1_pd(DELTAT);

    // 计算应力场 sigmaxx 和 sigmayy
    #pragma omp parallel for
    for (int j = 1; j < NY; j++) {
        // 确保内存对齐
        int i_start = 0;
        int i_end = ((NX - 1) / SIMD_WIDTH) * SIMD_WIDTH;
        
        for (int i = i_start; i < i_end; i += SIMD_WIDTH) {
            const int idx = i + j * NX;
            
            // 预取数据
            _mm_prefetch((const char*)&vx[idx + 16], _MM_HINT_T0);
            _mm_prefetch((const char*)&vy[idx + 16], _MM_HINT_T0);
            
            // 加载数据
            __m256d vx_vec = _mm256_loadu_pd(&vx[idx]);
            __m256d vx_next_vec = _mm256_loadu_pd(&vx[idx + 1]);
            __m256d vy_vec = _mm256_loadu_pd(&vy[idx]);
            __m256d vy_prev_vec = _mm256_loadu_pd(&vy[idx - NX]);
            
            // 计算速度梯度
            __m256d dvx_dx_vec = _mm256_mul_pd(
                _mm256_sub_pd(vx_next_vec, vx_vec),
                inv_dx_vec
            );
            __m256d dvy_dy_vec = _mm256_mul_pd(
                _mm256_sub_pd(vy_vec, vy_prev_vec),
                inv_dy_vec
            );
            
            // 加载PML系数
            __m256d b_x_half_vec = _mm256_loadu_pd(&b_x_half[i]);
            __m256d a_x_half_vec = _mm256_loadu_pd(&a_x_half[i]);
            __m256d b_y_vec = _mm256_set1_pd(b_y[j]);
            __m256d a_y_vec = _mm256_set1_pd(a_y[j]);
            
            // 更新记忆变量
            __m256d memory_dvx_dx_vec = _mm256_loadu_pd(&memory_dvx_dx[idx]);
            __m256d memory_dvy_dy_vec = _mm256_loadu_pd(&memory_dvy_dy[idx]);
            
            memory_dvx_dx_vec = _mm256_add_pd(
                _mm256_mul_pd(b_x_half_vec, memory_dvx_dx_vec),
                _mm256_mul_pd(a_x_half_vec, dvx_dx_vec)
            );
            memory_dvy_dy_vec = _mm256_add_pd(
                _mm256_mul_pd(b_y_vec, memory_dvy_dy_vec),
                _mm256_mul_pd(a_y_vec, dvy_dy_vec)
            );
            
            _mm256_storeu_pd(&memory_dvx_dx[idx], memory_dvx_dx_vec);
            _mm256_storeu_pd(&memory_dvy_dy[idx], memory_dvy_dy_vec);
            
            // 应用PML
            __m256d K_x_half_vec = _mm256_loadu_pd(&K_x_half[i]);
            __m256d K_y_vec = _mm256_set1_pd(K_y[j]);
            
            dvx_dx_vec = _mm256_add_pd(
                _mm256_div_pd(dvx_dx_vec, K_x_half_vec),
                memory_dvx_dx_vec
            );
            dvy_dy_vec = _mm256_add_pd(
                _mm256_div_pd(dvy_dy_vec, K_y_vec),
                memory_dvy_dy_vec
            );
            
            // 更新应力场
            __m256d c11_vec = _mm256_loadu_pd(&c11[idx]);
            __m256d c13_vec = _mm256_loadu_pd(&c13[idx]);
            __m256d c33_vec = _mm256_loadu_pd(&c33[idx]);
            
            __m256d sigmaxx_vec = _mm256_loadu_pd(&sigmaxx[idx]);
            __m256d sigmayy_vec = _mm256_loadu_pd(&sigmayy[idx]);
            
            sigmaxx_vec = _mm256_add_pd(
                sigmaxx_vec,
                _mm256_mul_pd(dt_vec,
                    _mm256_add_pd(
                        _mm256_mul_pd(c11_vec, dvx_dx_vec),
                        _mm256_mul_pd(c13_vec, dvy_dy_vec)
                    )
                )
            );
            
            sigmayy_vec = _mm256_add_pd(
                sigmayy_vec,
                _mm256_mul_pd(dt_vec,
                    _mm256_add_pd(
                        _mm256_mul_pd(c13_vec, dvx_dx_vec),
                        _mm256_mul_pd(c33_vec, dvy_dy_vec)
                    )
                )
            );
            
            _mm256_storeu_pd(&sigmaxx[idx], sigmaxx_vec);
            _mm256_storeu_pd(&sigmayy[idx], sigmayy_vec);
        }
        
        // 处理剩余元素
        for (int i = i_end; i < NX - 1; i++) {
            const int idx = i + j * NX;
            // 原始标量计算
            double value_dvx_dx = (vx[idx+1] - vx[idx]) * inv_dx;
            double value_dvy_dy = (vy[idx] - vy[idx-NX]) * inv_dy;
            
            memory_dvx_dx[idx] = b_x_half[i] * memory_dvx_dx[idx] + 
                                a_x_half[i] * value_dvx_dx;
            memory_dvy_dy[idx] = b_y[j] * memory_dvy_dy[idx] + 
                                a_y[j] * value_dvy_dy;
            
            value_dvx_dx = value_dvx_dx / K_x_half[i] + memory_dvx_dx[idx];
            value_dvy_dy = value_dvy_dy / K_y[j] + memory_dvy_dy[idx];
            
            sigmaxx[idx] += DELTAT * (c11[idx] * value_dvx_dx + c13[idx] * value_dvy_dy);
            sigmayy[idx] += DELTAT * (c13[idx] * value_dvx_dx + c33[idx] * value_dvy_dy);
        }
    }

    // 计算剪应力 sigmaxy
    #pragma omp parallel for
    for (int j = 0; j < NY - 1; j++) {
        int i_start = 1;
        int i_end = ((NX - 1) / SIMD_WIDTH) * SIMD_WIDTH;
        
        for (int i = i_start; i < i_end; i += SIMD_WIDTH) {
            const int idx = i + j * NX;
            
            _mm_prefetch((const char*)&vy[idx + 16], _MM_HINT_T0);
            _mm_prefetch((const char*)&vx[idx + 16], _MM_HINT_T0);
            
            // 计算速度梯度
            __m256d vy_vec = _mm256_loadu_pd(&vy[idx]);
            __m256d vy_prev_vec = _mm256_loadu_pd(&vy[idx-1]);
            __m256d vx_vec = _mm256_loadu_pd(&vx[idx]);
            __m256d vx_next_vec = _mm256_loadu_pd(&vx[idx+NX]);
            
            __m256d dvy_dx_vec = _mm256_mul_pd(
                _mm256_sub_pd(vy_vec, vy_prev_vec),
                inv_dx_vec
            );
            __m256d dvx_dy_vec = _mm256_mul_pd(
                _mm256_sub_pd(vx_next_vec, vx_vec),
                inv_dy_vec
            );
            
            // PML更新
            __m256d b_x_vec = _mm256_loadu_pd(&b_x[i]);
            __m256d a_x_vec = _mm256_loadu_pd(&a_x[i]);
            __m256d b_y_half_vec = _mm256_set1_pd(b_y_half[j]);
            __m256d a_y_half_vec = _mm256_set1_pd(a_y_half[j]);
            
            __m256d memory_dvy_dx_vec = _mm256_loadu_pd(&memory_dvy_dx[idx]);
            __m256d memory_dvx_dy_vec = _mm256_loadu_pd(&memory_dvx_dy[idx]);
            
            memory_dvy_dx_vec = _mm256_add_pd(
                _mm256_mul_pd(b_x_vec, memory_dvy_dx_vec),
                _mm256_mul_pd(a_x_vec, dvy_dx_vec)
            );
            memory_dvx_dy_vec = _mm256_add_pd(
                _mm256_mul_pd(b_y_half_vec, memory_dvx_dy_vec),
                _mm256_mul_pd(a_y_half_vec, dvx_dy_vec)
            );
            
            _mm256_storeu_pd(&memory_dvy_dx[idx], memory_dvy_dx_vec);
            _mm256_storeu_pd(&memory_dvx_dy[idx], memory_dvx_dy_vec);
            
            // 应用PML
            __m256d K_x_vec = _mm256_loadu_pd(&K_x[i]);
            __m256d K_y_half_vec = _mm256_set1_pd(K_y_half[j]);
            
            dvy_dx_vec = _mm256_add_pd(
                _mm256_div_pd(dvy_dx_vec, K_x_vec),
                memory_dvy_dx_vec
            );
            dvx_dy_vec = _mm256_add_pd(
                _mm256_div_pd(dvx_dy_vec, K_y_half_vec),
                memory_dvx_dy_vec
            );
            
            // 更新剪应力
            __m256d c44_vec = _mm256_loadu_pd(&c44[idx]);
            __m256d sigmaxy_vec = _mm256_loadu_pd(&sigmaxy[idx]);
            
            sigmaxy_vec = _mm256_add_pd(
                sigmaxy_vec,
                _mm256_mul_pd(
                    dt_vec,
                    _mm256_mul_pd(
                        c44_vec,
                        _mm256_add_pd(dvy_dx_vec, dvx_dy_vec)
                    )
                )
            );
            
            _mm256_storeu_pd(&sigmaxy[idx], sigmaxy_vec);
        }
        
        // 处理剩余元素
        for (int i = i_end; i < NX; i++) {
            const int idx = i + j * NX;
            double value_dvy_dx = (vy[idx] - vy[idx-1]) * inv_dx;
            double value_dvx_dy = (vx[idx+NX] - vx[idx]) * inv_dy;
            
            memory_dvy_dx[idx] = b_x[i] * memory_dvy_dx[idx] + a_x[i] * value_dvy_dx;
            memory_dvx_dy[idx] = b_y_half[j] * memory_dvx_dy[idx] + a_y_half[j] * value_dvx_dy;
            
            value_dvy_dx = value_dvy_dx / K_x[i] + memory_dvy_dx[idx];
            value_dvx_dy = value_dvx_dy / K_y_half[j] + memory_dvx_dy[idx];
            
            sigmaxy[idx] += DELTAT * c44[idx] * (value_dvy_dx + value_dvx_dy);
        }
    }

    // 计算x方向速度场
    #pragma omp parallel for
    for (int j = 1; j < NY; j++) {
        int i_start = 1;
        int i_end = ((NX - 1) / SIMD_WIDTH) * SIMD_WIDTH;
        
        for (int i = i_start; i < i_end; i += SIMD_WIDTH) {
            const int idx = i + j * NX;
            
            _mm_prefetch((const char*)&sigmaxx[idx + 16], _MM_HINT_T0);
            _mm_prefetch((const char*)&sigmaxy[idx + 16], _MM_HINT_T0);
            
            // 计算应力梯度
            __m256d sigmaxx_vec = _mm256_loadu_pd(&sigmaxx[idx]);
            __m256d sigmaxx_prev_vec = _mm256_loadu_pd(&sigmaxx[idx-1]);
            __m256d sigmaxy_vec = _mm256_loadu_pd(&sigmaxy[idx]);
            __m256d sigmaxy_prev_vec = _mm256_loadu_pd(&sigmaxy[idx-NX]);
            
            __m256d dsigmaxx_dx_vec = _mm256_mul_pd(
                _mm256_sub_pd(sigmaxx_vec, sigmaxx_prev_vec),
                inv_dx_vec
            );
            __m256d dsigmaxy_dy_vec = _mm256_mul_pd(
                _mm256_sub_pd(sigmaxy_vec, sigmaxy_prev_vec),
                inv_dy_vec
            );
            
            // PML更新
            __m256d b_x_vec = _mm256_loadu_pd(&b_x[i]);
            __m256d a_x_vec = _mm256_loadu_pd(&a_x[i]);
            __m256d b_y_vec = _mm256_set1_pd(b_y[j]);
            __m256d a_y_vec = _mm256_set1_pd(a_y[j]);
            
            __m256d memory_dsigmaxx_dx_vec = _mm256_loadu_pd(&memory_dsigmaxx_dx[idx]);
            __m256d memory_dsigmaxy_dy_vec = _mm256_loadu_pd(&memory_dsigmaxy_dy[idx]);
            
            memory_dsigmaxx_dx_vec = _mm256_add_pd(
                _mm256_mul_pd(b_x_vec, memory_dsigmaxx_dx_vec),
                _mm256_mul_pd(a_x_vec, dsigmaxx_dx_vec)
            );
            memory_dsigmaxy_dy_vec = _mm256_add_pd(
                _mm256_mul_pd(b_y_vec, memory_dsigmaxy_dy_vec),
                _mm256_mul_pd(a_y_vec, dsigmaxy_dy_vec)
            );
            
            _mm256_storeu_pd(&memory_dsigmaxx_dx[idx], memory_dsigmaxx_dx_vec);
            _mm256_storeu_pd(&memory_dsigmaxy_dy[idx], memory_dsigmaxy_dy_vec);
            
            // 应用PML
            __m256d K_x_vec = _mm256_loadu_pd(&K_x[i]);
            __m256d K_y_vec = _mm256_set1_pd(K_y[j]);
            
            dsigmaxx_dx_vec = _mm256_add_pd(
                _mm256_div_pd(dsigmaxx_dx_vec, K_x_vec),
                memory_dsigmaxx_dx_vec
            );
            dsigmaxy_dy_vec = _mm256_add_pd(
                _mm256_div_pd(dsigmaxy_dy_vec, K_y_vec),
                memory_dsigmaxy_dy_vec
            );
            
            // 更新速度场
            __m256d rho_vec = _mm256_loadu_pd(&rho[idx]);
            __m256d vx_vec = _mm256_loadu_pd(&vx[idx]);
            
            vx_vec = _mm256_add_pd(
                vx_vec,
                _mm256_mul_pd(
                    dt_vec,
                    _mm256_div_pd(
                        _mm256_add_pd(dsigmaxx_dx_vec, dsigmaxy_dy_vec),
                        rho_vec
                    )
                )
            );
            
            _mm256_storeu_pd(&vx[idx], vx_vec);
        }
        
        // 处理剩余元素
        for (int i = i_end; i < NX; i++) {
            const int idx = i + j * NX;
            double value_dsigmaxx_dx = (sigmaxx[idx] - sigmaxx[idx-1]) * inv_dx;
            double value_dsigmaxy_dy = (sigmaxy[idx] - sigmaxy[idx-NX]) * inv_dy;
            
            memory_dsigmaxx_dx[idx] = b_x[i] * memory_dsigmaxx_dx[idx] + a_x[i] * value_dsigmaxx_dx;
            memory_dsigmaxy_dy[idx] = b_y[j] * memory_dsigmaxy_dy[idx] + a_y[j] * value_dsigmaxy_dy;
            
            value_dsigmaxx_dx = value_dsigmaxx_dx / K_x[i] + memory_dsigmaxx_dx[idx];
            value_dsigmaxy_dy = value_dsigmaxy_dy / K_y[j] + memory_dsigmaxy_dy[idx];
            
            vx[idx] += (value_dsigmaxx_dx + value_dsigmaxy_dy) * DELTAT / rho[idx];
        }
    }

    // 计算y方向速度场
    #pragma omp parallel for
    for (int j = 0; j < NY - 1; j++) {
        int i_start = 0;
        int i_end = ((NX - 1) / SIMD_WIDTH) * SIMD_WIDTH;
        
        for (int i = i_start; i < i_end; i += SIMD_WIDTH) {
            const int idx = i + j * NX;
            
            _mm_prefetch((const char*)&sigmaxy[idx + 16], _MM_HINT_T0);
            _mm_prefetch((const char*)&sigmayy[idx + 16], _MM_HINT_T0);
            
            // 计算应力梯度
            __m256d sigmaxy_next_vec = _mm256_loadu_pd(&sigmaxy[idx+1]);
            __m256d sigmaxy_vec = _mm256_loadu_pd(&sigmaxy[idx]);
            __m256d sigmayy_next_vec = _mm256_loadu_pd(&sigmayy[idx+NX]);
            __m256d sigmayy_vec = _mm256_loadu_pd(&sigmayy[idx]);
            
            __m256d dsigmaxy_dx_vec = _mm256_mul_pd(
                _mm256_sub_pd(sigmaxy_next_vec, sigmaxy_vec),
                inv_dx_vec
            );
            __m256d dsigmayy_dy_vec = _mm256_mul_pd(
                _mm256_sub_pd(sigmayy_next_vec, sigmayy_vec),
                inv_dy_vec
            );
            
            // PML更新
            __m256d b_x_half_vec = _mm256_loadu_pd(&b_x_half[i]);
            __m256d a_x_half_vec = _mm256_loadu_pd(&a_x_half[i]);
            __m256d b_y_half_vec = _mm256_set1_pd(b_y_half[j]);
            __m256d a_y_half_vec = _mm256_set1_pd(a_y_half[j]);
            
            __m256d memory_dsigmaxy_dx_vec = _mm256_loadu_pd(&memory_dsigmaxy_dx[idx]);
            __m256d memory_dsigmayy_dy_vec = _mm256_loadu_pd(&memory_dsigmayy_dy[idx]);
            
            memory_dsigmaxy_dx_vec = _mm256_add_pd(
                _mm256_mul_pd(b_x_half_vec, memory_dsigmaxy_dx_vec),
                _mm256_mul_pd(a_x_half_vec, dsigmaxy_dx_vec)
            );
            memory_dsigmayy_dy_vec = _mm256_add_pd(
                _mm256_mul_pd(b_y_half_vec, memory_dsigmayy_dy_vec),
                _mm256_mul_pd(a_y_half_vec, dsigmayy_dy_vec)
            );
            
            _mm256_storeu_pd(&memory_dsigmaxy_dx[idx], memory_dsigmaxy_dx_vec);
            _mm256_storeu_pd(&memory_dsigmayy_dy[idx], memory_dsigmayy_dy_vec);
            
            // 应用PML
            __m256d K_x_half_vec = _mm256_loadu_pd(&K_x_half[i]);
            __m256d K_y_half_vec = _mm256_set1_pd(K_y_half[j]);
            
            dsigmaxy_dx_vec = _mm256_add_pd(
                _mm256_div_pd(dsigmaxy_dx_vec, K_x_half_vec),
                memory_dsigmaxy_dx_vec
            );
            dsigmayy_dy_vec = _mm256_add_pd(
                _mm256_div_pd(dsigmayy_dy_vec, K_y_half_vec),
                memory_dsigmayy_dy_vec
            );
            
            // 更新速度场
            __m256d rho_vec = _mm256_loadu_pd(&rho[idx]);
            __m256d vy_vec = _mm256_loadu_pd(&vy[idx]);
            
            vy_vec = _mm256_add_pd(
                vy_vec,
                _mm256_mul_pd(
                    dt_vec,
                    _mm256_div_pd(
                        _mm256_add_pd(dsigmaxy_dx_vec, dsigmayy_dy_vec),
                        rho_vec
                    )
                )
            );
            
            _mm256_storeu_pd(&vy[idx], vy_vec);
        }
        
        // 处理剩余元素
        for (int i = i_end; i < NX - 1; i++) {
            const int idx = i + j * NX;
            double value_dsigmaxy_dx = (sigmaxy[idx+1] - sigmaxy[idx]) * inv_dx;
            double value_dsigmayy_dy = (sigmayy[idx+NX] - sigmayy[idx]) * inv_dy;
            
            memory_dsigmaxy_dx[idx] = b_x_half[i] * memory_dsigmaxy_dx[idx] + a_x_half[i] * value_dsigmaxy_dx;
            memory_dsigmayy_dy[idx] = b_y_half[j] * memory_dsigmayy_dy[idx] + a_y_half[j] * value_dsigmayy_dy;
            
            value_dsigmaxy_dx = value_dsigmaxy_dx / K_x_half[i] + memory_dsigmaxy_dx[idx];
            value_dsigmayy_dy = value_dsigmayy_dy / K_y_half[j] + memory_dsigmayy_dy[idx];
            
            vy[idx] += (value_dsigmaxy_dx + value_dsigmayy_dy) * DELTAT / rho[idx];
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // 检查输入参数数量
    if (nrhs != 35) {
        mexErrMsgIdAndTxt("MyToolbox:VTI_WaveFieldSolver_SIMD:nrhs",
                         "Need 35 input arguments.");
    }
    
    // 检查输出参数数量
    if (nlhs != 2) {
        mexErrMsgIdAndTxt("MyToolbox:VTI_WaveFieldSolver_SIMD:nlhs",
                         "Two outputs required.");
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

    // 网格参数
    double DELTAX = mxGetScalar(prhs[30]);
    double DELTAY = mxGetScalar(prhs[31]);
    double DELTAT = mxGetScalar(prhs[32]);
    int NX = static_cast<int>(mxGetScalar(prhs[33]));
    int NY = static_cast<int>(mxGetScalar(prhs[34]));
    
    // 设置OpenMP线程数
    int num_threads = omp_get_num_procs();  // 获取CPU核心数
    omp_set_num_threads(num_threads);       // 设置线程数
    
    // 调用计算函数
    compute_wave_propagation_simd(
        vx, vy, sigmaxx, sigmayy, sigmaxy,
        memory_dvx_dx, memory_dvy_dy, memory_dvy_dx, memory_dvx_dy,
        memory_dsigmaxx_dx, memory_dsigmaxy_dy, memory_dsigmaxy_dx, memory_dsigmayy_dy,
        c11, c13, c33, c44, rho,
        b_x, b_y, b_x_half, b_y_half,
        a_x, a_y, a_x_half, a_y_half,
        K_x, K_y, K_x_half, K_y_half,
        DELTAX, DELTAY, DELTAT, NX, NY);
    
    // 创建输出矩阵并复制结果
    plhs[0] = mxCreateDoubleMatrix(NX, NY, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(NX, NY, mxREAL);
    
    // 复制结果到输出矩阵
    memcpy(mxGetPr(plhs[0]), vx, NX * NY * sizeof(double));
    memcpy(mxGetPr(plhs[1]), vy, NX * NY * sizeof(double));
} 