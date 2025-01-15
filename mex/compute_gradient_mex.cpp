#include "mex.h"
#include "matrix.h"
#include <cmath>
#include <cstring>

// 计算水平方向导数
void compute_dx(double* dx, const double* field, const double deltax, 
               const mwSize nx, const mwSize ny) {
    const double c1 = 9.0/8.0;
    const double c2 = -1.0/24.0;
    
    // 内部点（列优先）
    for(mwSize j = 2; j < nx-2; j++) {
        for(mwSize i = 0; i < ny; i++) {
            dx[i + j*ny] = (c1*(field[i + (j+1)*ny] - field[i + (j-1)*ny]) +
                           c2*(field[i + (j+2)*ny] - field[i + (j-2)*ny])) / deltax;
        }
    }
    
    // 边界处理（列优先）
    for(mwSize i = 0; i < ny; i++) {
        // 左边界（第1,2列）
        dx[i + 0*ny] = (-3.0*field[i + 0*ny] + 4.0*field[i + 1*ny] - field[i + 2*ny]) / (2.0*deltax);
        dx[i + 1*ny] = (-3.0*field[i + 1*ny] + 4.0*field[i + 2*ny] - field[i + 3*ny]) / (2.0*deltax);
        
        // 右边界（倒数第2,1列）
        dx[i + (nx-2)*ny] = (3.0*field[i + (nx-2)*ny] - 4.0*field[i + (nx-3)*ny] + field[i + (nx-4)*ny]) / (2.0*deltax);
        dx[i + (nx-1)*ny] = (3.0*field[i + (nx-1)*ny] - 4.0*field[i + (nx-2)*ny] + field[i + (nx-3)*ny]) / (2.0*deltax);
    }
}

// 计算垂直方向导数
void compute_dy(double* dy, const double* field, const double deltay, 
               const mwSize nx, const mwSize ny) {
    const double c1 = 9.0/8.0;
    const double c2 = -1.0/24.0;
    
    // 内部点（列优先）
    for(mwSize j = 0; j < nx; j++) {
        for(mwSize i = 2; i < ny-2; i++) {
            // 四阶中心差分
            dy[i + j*ny] = (c1*(field[(i+1) + j*ny] - field[(i-1) + j*ny]) +
                           c2*(field[(i+2) + j*ny] - field[(i-2) + j*ny])) / deltay;
        }
    }
    
    // 处理边界（二阶差分）
    for(mwSize j = 0; j < nx; j++) {
        // 上边界（第1,2行）
        dy[0 + j*ny] = (-3.0*field[0 + j*ny] + 4.0*field[1 + j*ny] - field[2 + j*ny]) / (2.0*deltay);
        dy[1 + j*ny] = (-3.0*field[1 + j*ny] + 4.0*field[2 + j*ny] - field[3 + j*ny]) / (2.0*deltay);
        
        // 下边界（倒数第2,1行）
        dy[(ny-2) + j*ny] = (3.0*field[(ny-2) + j*ny] - 4.0*field[(ny-3) + j*ny] + field[(ny-4) + j*ny]) / (2.0*deltay);
        dy[(ny-1) + j*ny] = (3.0*field[(ny-1) + j*ny] - 4.0*field[(ny-2) + j*ny] + field[(ny-3) + j*ny]) / (2.0*deltay);
    }
}

// MEX函数入口点
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // 1. 检查输入参数
    if (nrhs != 3) {
        mexErrMsgIdAndTxt("MyToolbox:compute_gradient:nrhs",
                         "Need 3 input arguments: field, deltax, deltay");
    }
    if (nlhs != 2) {
        mexErrMsgIdAndTxt("MyToolbox:compute_gradient:nlhs",
                         "Need 2 output arguments: dx, dy");
    }
    
    // 2. 获取输入数据和维度
    const mwSize nx = mxGetM(prhs[0]);  
    const mwSize ny = mxGetN(prhs[0]);  
    const double* field = mxGetPr(prhs[0]);
    const double deltax = mxGetScalar(prhs[1]);
    const double deltay = mxGetScalar(prhs[2]);
    
    // 3. 创建输出矩阵
    plhs[0] = mxCreateDoubleMatrix(nx, ny, mxREAL);  
    plhs[1] = mxCreateDoubleMatrix(nx, ny, mxREAL);  
    
    // 4. 计算导数
    compute_dx(mxGetPr(plhs[0]), field, deltax, nx, ny);
    compute_dy(mxGetPr(plhs[1]), field, deltay, nx, ny);
} 