#include "mex.h"
#include "matrix.h"
#include <cmath>
#include <vector>
#include <omp.h>

// 辅助函数：计算最大绝对值
static double computeMaxAbs(const double* array, size_t size) {
    double max_val = 0.0;
    #pragma omp parallel
    {
        double local_max = 0.0;
        #pragma omp for
        for (int i = 0; i < static_cast<int>(size); i++) {
            double abs_val = std::abs(array[i]);
            if (abs_val > local_max) local_max = abs_val;
        }
        #pragma omp critical
        {
            if (local_max > max_val) max_val = local_max;
        }
    }
    return max_val;
}

// 辅助函数：计算空间梯度
static void computeGradient(const double* field, double* dx, double* dy, 
                    size_t nx, size_t ny, double deltax, double deltay) {
    // 二阶中心差分实现
    #pragma omp parallel for
    for (int j = 0; j < static_cast<int>(ny); j++) {
        for (size_t i = 1; i < nx-1; i++) {
            dx[i + j*nx] = (field[i+1 + j*nx] - field[i-1 + j*nx]) / (2.0 * deltax);
        }
        // 处理边界点
        dx[0 + j*nx] = (field[1 + j*nx] - field[0 + j*nx]) / deltax;
        dx[(nx-1) + j*nx] = (field[(nx-1) + j*nx] - field[(nx-2) + j*nx]) / deltax;
    }
    
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(nx); i++) {
        for (size_t j = 1; j < ny-1; j++) {
            dy[i + j*nx] = (field[i + (j+1)*nx] - field[i + (j-1)*nx]) / (2.0 * deltay);
        }
        // 处理边界点
        dy[i + 0*nx] = (field[i + 1*nx] - field[i + 0*nx]) / deltay;
        dy[i + (ny-1)*nx] = (field[i + (ny-1)*nx] - field[i + (ny-2)*nx]) / deltay;
    }
}

#ifdef __cplusplus
extern "C" {
#endif

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

#ifdef __cplusplus
}
#endif

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // 检查输入参数
    if (nrhs != 4) {
        mexErrMsgIdAndTxt("correlate_wavefields_mex:invalidNumInputs",
            "Need 4 inputs: forward_wavefield, adjoint_wavefield, dt, params");
    }
    
    // 设置OpenMP线程数
    int num_threads = omp_get_num_procs();
    omp_set_num_threads(num_threads);
    
    // 获取输入数据
    const mxArray* forward_vx_array = mxGetField(prhs[0], 0, "vx");
    const mxArray* forward_vy_array = mxGetField(prhs[0], 0, "vy");
    const mxArray* adjoint_vx_array = mxGetField(prhs[1], 0, "vx");
    const mxArray* adjoint_vy_array = mxGetField(prhs[1], 0, "vy");
    
    double dt = mxGetScalar(prhs[2]);
    double deltax = mxGetScalar(mxGetField(prhs[3], 0, "DELTAX"));
    double deltay = mxGetScalar(mxGetField(prhs[3], 0, "DELTAY"));
    
    // 获取维度信息
    const mwSize* dims = mxGetDimensions(forward_vx_array);
    size_t nx = dims[0];
    size_t ny = dims[1];
    size_t nt = dims[2];
    
    // 创建输出结构体
    const char* fieldnames[] = {"c11", "c13", "c33", "c44", "rho"};
    plhs[0] = mxCreateStructMatrix(1, 1, 5, fieldnames);
    
    mxArray* gradient_c11 = mxCreateDoubleMatrix(nx, ny, mxREAL);
    mxArray* gradient_c13 = mxCreateDoubleMatrix(nx, ny, mxREAL);
    mxArray* gradient_c33 = mxCreateDoubleMatrix(nx, ny, mxREAL);
    mxArray* gradient_c44 = mxCreateDoubleMatrix(nx, ny, mxREAL);
    mxArray* gradient_rho = mxCreateDoubleMatrix(nx, ny, mxREAL);
    
    // 获取数据指针
    double* c11_ptr = mxGetPr(gradient_c11);
    double* c13_ptr = mxGetPr(gradient_c13);
    double* c33_ptr = mxGetPr(gradient_c33);
    double* c44_ptr = mxGetPr(gradient_c44);
    double* rho_ptr = mxGetPr(gradient_rho);
    
    // 分配临时数组
    std::vector<double> dvx_dx(nx*ny);
    std::vector<double> dvx_dy(nx*ny);
    std::vector<double> dvy_dx(nx*ny);
    std::vector<double> dvy_dy(nx*ny);
    std::vector<double> dadj_vx_dx(nx*ny);
    std::vector<double> dadj_vx_dy(nx*ny);
    std::vector<double> dadj_vy_dx(nx*ny);
    std::vector<double> dadj_vy_dy(nx*ny);
    
    const double* fwd_vx_all = mxGetPr(forward_vx_array);
    const double* fwd_vy_all = mxGetPr(forward_vy_array);
    
    // 主循环：时间步迭代
    for (size_t it = 0; it < nt; it++) {
        // 获取当前时间步的波场数据
        const double* fwd_vx = fwd_vx_all + it*nx*ny;
        const double* fwd_vy = fwd_vy_all + it*nx*ny;
        const double* adj_vx = mxGetPr(adjoint_vx_array) + it*nx*ny;
        const double* adj_vy = mxGetPr(adjoint_vy_array) + it*nx*ny;
        
        // 计算空间导数
        computeGradient(fwd_vx, dvx_dx.data(), dvx_dy.data(), nx, ny, deltax, deltay);
        computeGradient(fwd_vy, dvy_dx.data(), dvy_dy.data(), nx, ny, deltax, deltay);
        computeGradient(adj_vx, dadj_vx_dx.data(), dadj_vx_dy.data(), nx, ny, deltax, deltay);
        computeGradient(adj_vy, dadj_vy_dx.data(), dadj_vy_dy.data(), nx, ny, deltax, deltay);
        
        // 更新梯度
        #pragma omp parallel for
        for (int idx = 0; idx < static_cast<int>(nx*ny); idx++) {
            double dv_dt_x, dv_dt_y;
            
            if (it == 0) {
                dv_dt_x = (fwd_vx_all[idx + (it+1)*nx*ny] - fwd_vx_all[idx + it*nx*ny]) / dt;
                dv_dt_y = (fwd_vy_all[idx + (it+1)*nx*ny] - fwd_vy_all[idx + it*nx*ny]) / dt;
            } else if (it == nt-1) {
                dv_dt_x = (fwd_vx_all[idx + it*nx*ny] - fwd_vx_all[idx + (it-1)*nx*ny]) / dt;
                dv_dt_y = (fwd_vy_all[idx + it*nx*ny] - fwd_vy_all[idx + (it-1)*nx*ny]) / dt;
            } else {
                dv_dt_x = (fwd_vx_all[idx + (it+1)*nx*ny] - fwd_vx_all[idx + (it-1)*nx*ny]) / (2.0*dt);
                dv_dt_y = (fwd_vy_all[idx + (it+1)*nx*ny] - fwd_vy_all[idx + (it-1)*nx*ny]) / (2.0*dt);
            }
            
            #pragma omp atomic
            c11_ptr[idx] -= dvx_dx[idx] * dadj_vx_dx[idx] * dt;
            
            #pragma omp atomic
            c13_ptr[idx] -= (dadj_vx_dx[idx] * dvy_dy[idx] + 
                            dadj_vy_dy[idx] * dvx_dx[idx]) * 2.0 * dt;
            
            #pragma omp atomic
            c33_ptr[idx] -= dvy_dy[idx] * dadj_vy_dy[idx] * dt;
            
            #pragma omp atomic
            c44_ptr[idx] -= (dvx_dy[idx] + dvy_dx[idx]) * 
                           (dadj_vx_dy[idx] + dadj_vy_dx[idx]) * dt;
            
            #pragma omp atomic
            rho_ptr[idx] -= (dadj_vx_dx[idx] * dv_dt_x + 
                            dadj_vy_dy[idx] * dv_dt_y) * dt;
        }
        
        // 进度输出
        if ((it + 1) % 100 == 0) {
            mexPrintf("Time step %zu:\n", it + 1);
            mexPrintf("C11 max gradient: %e\n", computeMaxAbs(c11_ptr, nx*ny));
            mexPrintf("C13 max gradient: %e\n", computeMaxAbs(c13_ptr, nx*ny));
            mexPrintf("C33 max gradient: %e\n", computeMaxAbs(c33_ptr, nx*ny));
            mexPrintf("C44 max gradient: %e\n", computeMaxAbs(c44_ptr, nx*ny));
            mexPrintf("Rho max gradient: %e\n", computeMaxAbs(rho_ptr, nx*ny));
            mexEvalString("drawnow;");
        }
    }
    
    // 设置输出结构体字段
    mxSetField(plhs[0], 0, "c11", gradient_c11);
    mxSetField(plhs[0], 0, "c13", gradient_c13);
    mxSetField(plhs[0], 0, "c33", gradient_c33);
    mxSetField(plhs[0], 0, "c44", gradient_c44);
    mxSetField(plhs[0], 0, "rho", gradient_rho);
}