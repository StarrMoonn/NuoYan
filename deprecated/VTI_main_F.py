import numpy as np
import matplotlib.pyplot as plt
import os
import cupy as cp

cuda_code = r'''
extern "C" __global__ void compute_stress(
    const double* vx, const double* vy,
    double* sigmaxx, double* sigmayy, double* sigmaxy,
    double* memory_dvx_dx, double* memory_dvy_dy,
    double* memory_dvy_dx, double* memory_dvx_dy,
    const double* b_x_half, const double* b_y, const double* a_x_half, const double* a_y,
    const double* b_x, const double* b_y_half, const double* a_x, const double* a_y_half,
    const double* K_x_half, const double* K_y, const double* K_x, const double* K_y_half,
    double DELTAX, double DELTAY, double DELTAT,
    double c11, double c12, double c22, double c33,
    int NX, int NY)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // 网格 i 索引
    int j = blockIdx.y * blockDim.y + threadIdx.y; // 网格 j 索引
    
    if (j >= 1 && j < NY && i >= 0 && i < NX - 1) {
        // 第一部分：正应力分量
        double value_dvx_dx = (vx[(i + 1) * NY + j] - vx[i * NY + j]) / DELTAX;
        double value_dvy_dy = (vy[i * NY + j] - vy[i * NY + (j - 1)]) / DELTAY;

        memory_dvx_dx[i * NY + j] = b_x_half[i] * memory_dvx_dx[i * NY + j] +
                                    a_x_half[i] * value_dvx_dx;
        memory_dvy_dy[i * NY + j] = b_y[j] * memory_dvy_dy[i * NY + j] +
                                    a_y[j] * value_dvy_dy;

        value_dvx_dx = value_dvx_dx / K_x_half[i] + memory_dvx_dx[i * NY + j];
        value_dvy_dy = value_dvy_dy / K_y[j] + memory_dvy_dy[i * NY + j];

        sigmaxx[i * NY + j] += (c11 * value_dvx_dx + c12 * value_dvy_dy) * DELTAT;
        sigmayy[i * NY + j] += (c12 * value_dvx_dx + c22 * value_dvy_dy) * DELTAT;
    }

    if (j >= 0 && j < NY - 1 && i >= 1 && i < NX) {
        // 第二部分：剪切应力分量
        double value_dvy_dx = (vy[i * NY + j] - vy[(i - 1) * NY + j]) / DELTAX;
        double value_dvx_dy = (vx[i * NY + (j + 1)] - vx[i * NY + j]) / DELTAY;

        memory_dvy_dx[i * NY + j] = b_x[i] * memory_dvy_dx[i * NY + j] +
                                    a_x[i] * value_dvy_dx;
        memory_dvx_dy[i * NY + j] = b_y_half[j] * memory_dvx_dy[i * NY + j] +
                                    a_y_half[j] * value_dvx_dy;

        value_dvy_dx = value_dvy_dx / K_x[i] + memory_dvy_dx[i * NY + j];
        value_dvx_dy = value_dvx_dy / K_y_half[j] + memory_dvx_dy[i * NY + j];

        sigmaxy[i * NY + j] += c33 * (value_dvy_dx + value_dvx_dy) * DELTAT;
    }
}

extern "C" __global__ void compute_velocity(
    const double* sigmaxx, const double* sigmayy, const double* sigmaxy,
    double* vx, double* vy,
    double* memory_dsigmaxx_dx, double* memory_dsigmaxy_dy,
    double* memory_dsigmaxy_dx, double* memory_dsigmayy_dy,
    const double* b_x, const double* b_y, const double* a_x, const double* a_y,
    const double* b_x_half, const double* b_y_half, const double* a_x_half, const double* a_y_half,
    const double* K_x, const double* K_y, const double* K_x_half, const double* K_y_half,
    double DELTAX, double DELTAY, double DELTAT, double rho,
    int NX, int NY)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= 1 && j < NY && i >= 1 && i < NX) {
        double value_dsigmaxx_dx = (sigmaxx[i * NY + j] - sigmaxx[(i - 1) * NY + j]) / DELTAX;
        double value_dsigmaxy_dy = (sigmaxy[i * NY + j] - sigmaxy[i * NY + (j - 1)]) / DELTAY;

        memory_dsigmaxx_dx[i * NY + j] = b_x[i] * memory_dsigmaxx_dx[i * NY + j] +
                                         a_x[i] * value_dsigmaxx_dx;
        memory_dsigmaxy_dy[i * NY + j] = b_y[j] * memory_dsigmaxy_dy[i * NY + j] +
                                         a_y[j] * value_dsigmaxy_dy;

        value_dsigmaxx_dx = value_dsigmaxx_dx / K_x[i] + memory_dsigmaxx_dx[i * NY + j];
        value_dsigmaxy_dy = value_dsigmaxy_dy / K_y[j] + memory_dsigmaxy_dy[i * NY + j];

        vx[i * NY + j] += (value_dsigmaxx_dx + value_dsigmaxy_dy) * DELTAT / rho;
    }

    if (j >= 0 && j < NY - 1 && i >= 0 && i < NX - 1) {
        double value_dsigmaxy_dx = (sigmaxy[(i + 1) * NY + j] - sigmaxy[i * NY + j]) / DELTAX;
        double value_dsigmayy_dy = (sigmayy[i * NY + (j + 1)] - sigmayy[i * NY + j]) / DELTAY;

        memory_dsigmaxy_dx[i * NY + j] = b_x_half[i] * memory_dsigmaxy_dx[i * NY + j] +
                                         a_x_half[i] * value_dsigmaxy_dx;
        memory_dsigmayy_dy[i * NY + j] = b_y_half[j] * memory_dsigmayy_dy[i * NY + j] +
                                         a_y_half[j] * value_dsigmayy_dy;

        value_dsigmaxy_dx = value_dsigmaxy_dx / K_x_half[i] + memory_dsigmaxy_dx[i * NY + j];
        value_dsigmayy_dy = value_dsigmayy_dy / K_y_half[j] + memory_dsigmayy_dy[i * NY + j];

        vy[i * NY + j] += (value_dsigmaxy_dx + value_dsigmayy_dy) * DELTAT / rho;
    }
}

'''


module = cp.RawModule(code=cuda_code)
compute_stress = module.get_function("compute_stress")
compute_velocity = module.get_function("compute_velocity")


class SeismicCPML2DAniso:
    def __init__(self):
        # Grid parameters
        self.NX = 401     
        self.NY = 401
        
        # Grid spacing
        self.DELTAX = 5 
        self.DELTAY = 0.0625e-2
        
        # PML flags and parameters  
        self.USE_PML_XMIN = True
        self.USE_PML_XMAX = True
        self.USE_PML_YMIN = True
        self.USE_PML_YMAX = True
        self.NPOINTS_PML = 10
        
        # Material properties (Model I from Becache, Fauqueux and Joly)
        self.scale_aniso = 1.0e10     # 各向异性参数缩放因子
        self.c11 = 4.0 * self.scale_aniso    # VTI介质刚度系数c11
        self.c12 = 3.8 * self.scale_aniso    # VTI介质刚度系数c12
        self.c22 = 20.0 * self.scale_aniso   # VTI介质刚度系数c22
        self.c33 = 2.0 * self.scale_aniso    # VTI介质刚度系数c33
        self.rho = 4000.0             # 介质密度(kg/m³)
        self.f0 = 200.0e3             # 震源主频(Hz)
        
        # Time stepping parameters
        self.NSTEP = 3000
        self.DELTAT = 50.0e-9         # 时间步长(s)
        
        # 震源参数设置
        self.t0 = 1.20/self.f0        # 时间延迟
        self.factor = 1.0e7           # 震源振幅因子
        self.ISOURCE = self.NX // 2   # 震源x位置(网格点)
        self.JSOURCE = self.NY // 2   # 震源y位置(网格点)
        self.xsource = (self.ISOURCE - 1) * self.DELTAX  # 震源实际x坐标
        self.ysource = (self.JSOURCE - 1) * self.DELTAY  # 震源实际y坐标
        self.ANGLE_FORCE = 0.0        # 震源力的方向角度(度)
        
        # 检波器参数设置
        self.NREC = 50                # 检波器数量
        self.first_rec_x = 100        # 第一个检波器x位置
        self.first_rec_z = 50         # 第一个检波器z位置
        self.rec_dx = 4               # 检波器x方向间距
        self.rec_dz = 0               # 检波器z方向间距
        
        # 初始化检波器数组
        self.rec_x = np.zeros(self.NREC, dtype=np.int32)  # 检波器x坐标数组
        self.rec_z = np.zeros(self.NREC, dtype=np.int32)  # 检波器z坐标数组
        
        # 初始化地震记录数组
        self.seismogram_vx = np.zeros((self.NSTEP, self.NREC))  # x方向速度记录
        self.seismogram_vz = np.zeros((self.NSTEP, self.NREC))  # z方向速度记录
        
        # 显示参数
        self.IT_DISPLAY = 100         # 波场快照输出间隔
        
        # 常量定义
        self.PI = cp.pi               # 圆周率
        self.DEGREES_TO_RADIANS = self.PI / 180.0  # 角度转弧度系数
        self.ZERO = cp.float64(0.0)   # 零值常量
        self.HUGEVAL = cp.float64(1.0e+30)  # 大数值常量
        self.STABILITY_THRESHOLD = cp.float64(1.0e+25)  # 稳定性阈值
        
        # PML参数
        self.NPOWER = cp.float64(2.0)  # PML衰减函数的幂次
        self.K_MAX_PML = cp.float64(1.0)  # PML最大拉伸系数
        self.ALPHA_MAX_PML = cp.float64(2.0 * self.PI * (self.f0/2.0))  # PML最大频率调制系数
        
        # 初始化计算所需数组
        self.initialize_arrays()
        
        # 设置检波器位置
        self.setup_receivers()
        
        # 定义多炮记录数组
        self.NSHOT = 1  # 炮数(可以修改)
        
        # 创建两个三维数组来存储多炮记录
        # 维度: (炮点数, 时间步数, 检波器数)
        self.shot_records_vx = np.zeros((self.NSHOT, self.NSTEP, self.NREC))
        self.shot_records_vz = np.zeros((self.NSHOT, self.NSTEP, self.NREC))
        
        # 数组维度说明:
        # 第一维 (NSHOT): 不同的炮点位置
        # 第二维 (NSTEP): 时间采样点
        # 第三维 (NREC):  检波器通道数

        # 添加输出目录设置
        self.output_dir = 'output'
        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def initialize_arrays(self):
        """初始化模拟所需的所有GPU数组"""
        # 主要场变量数组
        self.vx = cp.zeros((self.NX, self.NY), dtype=cp.float64)      # x方向速度分量
        self.vy = cp.zeros((self.NX, self.NY), dtype=cp.float64)      # y方向速度分量
        self.sigmaxx = cp.zeros((self.NX, self.NY), dtype=cp.float64) # xx方向应力分量
        self.sigmayy = cp.zeros((self.NX, self.NY), dtype=cp.float64) # yy方向应力分量
        self.sigmaxy = cp.zeros((self.NX, self.NY), dtype=cp.float64) # xy方向应力分量
        
        # PML区域的记忆变量数组
        # 速度导数的记忆变量
        self.memory_dvx_dx = cp.zeros((self.NX, self.NY), dtype=cp.float64)  # vx对x的导数记忆变量
        self.memory_dvx_dy = cp.zeros((self.NX, self.NY), dtype=cp.float64)  # vx对y的导数记忆变量
        self.memory_dvy_dx = cp.zeros((self.NX, self.NY), dtype=cp.float64)  # vy对x的导数记忆变量
        self.memory_dvy_dy = cp.zeros((self.NX, self.NY), dtype=cp.float64)  # vy对y的导数记忆变量
        
        # 应力导数的记忆变量
        self.memory_dsigmaxx_dx = cp.zeros((self.NX, self.NY), dtype=cp.float64)  # σxx对x的导数记忆变量
        self.memory_dsigmayy_dy = cp.zeros((self.NX, self.NY), dtype=cp.float64)  # σyy对y的导数记忆变量
        self.memory_dsigmaxy_dx = cp.zeros((self.NX, self.NY), dtype=cp.float64)  # σxy对x的导数记忆变量
        self.memory_dsigmaxy_dy = cp.zeros((self.NX, self.NY), dtype=cp.float64)  # σxy对y的导数记忆变量
        
        # PML衰减参数的一维数组
        # x方向的衰减参数
        self.d_x = cp.zeros(self.NX, dtype=cp.float64)        # 衰减函数
        self.d_x_half = cp.zeros(self.NX, dtype=cp.float64)   # 半网格点衰减函数
        self.K_x = cp.ones(self.NX, dtype=cp.float64)         # 拉伸函数
        self.K_x_half = cp.ones(self.NX, dtype=cp.float64)    # 半网格点拉伸函数
        self.alpha_x = cp.zeros(self.NX, dtype=cp.float64)    # 频率转换函数
        self.alpha_x_half = cp.zeros(self.NX, dtype=cp.float64)
        self.a_x = cp.zeros(self.NX, dtype=cp.float64)        # CPML更新系数a
        self.a_x_half = cp.zeros(self.NX, dtype=cp.float64)
        self.b_x = cp.zeros(self.NX, dtype=cp.float64)        # CPML更新系数b
        self.b_x_half = cp.zeros(self.NX, dtype=cp.float64)
        
        # y方向的衰减参数（与x方向类似）
        self.d_y = cp.zeros(self.NY, dtype=cp.float64)
        self.d_y_half = cp.zeros(self.NY, dtype=cp.float64)
        self.K_y = cp.ones(self.NY, dtype=cp.float64)
        self.K_y_half = cp.ones(self.NY, dtype=cp.float64)
        self.alpha_y = cp.zeros(self.NY, dtype=cp.float64)
        self.alpha_y_half = cp.zeros(self.NY, dtype=cp.float64)
        self.a_y = cp.zeros(self.NY, dtype=cp.float64)
        self.a_y_half = cp.zeros(self.NY, dtype=cp.float64)
        self.b_y = cp.zeros(self.NY, dtype=cp.float64)
        self.b_y_half = cp.zeros(self.NY, dtype=cp.float64)

    def add_source(self, it):
        """添加震源（在指定网格点处添加力矢量）"""
        # 计算高斯函数的参数
        a = self.PI * self.PI * self.f0 * self.f0    # 高斯函数的频率参数
        t = (it-1) * self.DELTAT                     # 当前时刻
        
        # 计算高斯函数的一阶导数作为震源时间函数
        source_term = -self.factor * 2.0 * a * (t-self.t0) * cp.exp(-a*(t-self.t0)**2)
        
        # 根据震源角度分解力矢量到x和y方向
        force_x = cp.sin(self.ANGLE_FORCE * self.DEGREES_TO_RADIANS) * source_term  # x方向分量
        force_y = cp.cos(self.ANGLE_FORCE * self.DEGREES_TO_RADIANS) * source_term  # y方向分量
        
        # 获取震源位置
        i = self.ISOURCE  # 震源x坐标
        j = self.JSOURCE  # 震源y坐标
        
        # 将力添加到速度场中
        self.vx[i,j] += force_x * self.DELTAT / self.rho  # 更新x方向速度
        self.vy[i,j] += force_y * self.DELTAT / self.rho  # 更新y方向速度

    def setup_receivers(self):
        """设置检波器位置"""
        # 根据初始位置和间距计算每个检波器的坐标
        for i in range(self.NREC):
            self.rec_x[i] = self.first_rec_x + i * self.rec_dx    # 计算x方向位置
            self.rec_z[i] = self.first_rec_z + i * self.rec_dz    # 计算z方向位置
        
        # 检查检波器位置是否在计算网格范围内
        if np.any(self.rec_x >= self.NX) or np.any(self.rec_z >= self.NY):
            raise ValueError("检波器位置超出了计算网格范围")

    def setup_pml_boundary(self):
        """设置PML边界条件并检查稳定性"""
        
        # 检查各向异性材料PML模型的稳定性条件
        aniso_stability_criterion = ((self.c12 + self.c33)**2 - self.c11*(self.c22-self.c33)) * \
                                   ((self.c12 + self.c33)**2 + self.c33*(self.c22-self.c33))
        print(f'Becache等人2003年提出的PML各向异性稳定性判据 = {aniso_stability_criterion}')
        if aniso_stability_criterion > 0.0 and (self.USE_PML_XMIN or self.USE_PML_XMAX or 
                                              self.USE_PML_YMIN or self.USE_PML_YMAX):
            print('警告：对于该各向异性材料，PML模型在条件1下在数学上本质上不稳定')
        
        # 检查第二个稳定性条件
        aniso2 = (self.c12 + 2*self.c33)**2 - self.c11*self.c22
        print(f'Becache等人2003年提出的PML aniso2稳定性判据 = {aniso2}')
        if aniso2 > 0.0 and (self.USE_PML_XMIN or self.USE_PML_XMAX or 
                             self.USE_PML_YMIN or self.USE_PML_YMAX):
            print('警告：对于该各向异性材料，PML模型在条件2下在数学上本质上不稳定')
        
        # 检查第三个稳定性条件
        aniso3 = (self.c12 + self.c33)**2 - self.c11*self.c22 - self.c33**2
        print(f'Becache等人2003年提出的PML aniso3稳定性判据 = {aniso3}')
        if aniso3 > 0.0 and (self.USE_PML_XMIN or self.USE_PML_XMAX or 
                             self.USE_PML_YMIN or self.USE_PML_YMAX):
            print('警告：对于该各向异性材料，PML模型在条件3下在数学上本质上不稳定')

    def setup_pml_boundary_x(self):
        """设置x方向的PML边界条件"""
        # 计算准P波最大速度，用于d0计算
        quasi_cp_max = cp.maximum(cp.sqrt(self.c22/self.rho), cp.sqrt(self.c11/self.rho))
        
        # 定义PML区域的吸收层厚度
        thickness_PML_x = self.NPOINTS_PML * self.DELTAX
        
        # 设置反射系数（控制PML的吸收效果）
        Rcoef = cp.float64(0.001)
        
        # 检查NPOWER值的有效性
        if self.NPOWER < 1:
            raise ValueError('NPOWER必须大于1')
        
        # 计算衰减系数d0
        d0_x = -(self.NPOWER + 1) * quasi_cp_max * cp.log(Rcoef) / (2.0 * thickness_PML_x)
        print(f'd0_x = {d0_x}')
        
        # 设置衰减区域的边界位置
        xoriginleft = thickness_PML_x                           # 左边界位置
        xoriginright = (self.NX-1)*self.DELTAX - thickness_PML_x  # 右边界位置
        
        # 创建x方向的网格点坐标数组
        x_vals = cp.arange(self.NX, dtype=cp.float64) * self.DELTAX      # 整数网格点
        x_vals_half = x_vals + self.DELTAX/2.0                           # 半网格点
        
        # 处理左边界PML
        if self.USE_PML_XMIN:
            # 计算在PML区域内的位置
            abscissa_in_PML = xoriginleft - x_vals
            mask = abscissa_in_PML >= 0.0
            # 归一化位置（0到1之间）
            abscissa_normalized = cp.where(mask, abscissa_in_PML / thickness_PML_x, 0.0)
            
            # 设置整数网格点的PML参数
            self.d_x = cp.where(mask, d0_x * abscissa_normalized**self.NPOWER, self.d_x)           # 衰减函数
            self.K_x = cp.where(mask, 1.0 + (self.K_MAX_PML - 1.0) * abscissa_normalized**self.NPOWER, self.K_x)  # 拉伸函数
            self.alpha_x = cp.where(mask, self.ALPHA_MAX_PML * (1.0 - abscissa_normalized), self.alpha_x)  # 频率调制函数
            
            # 设置半网格点的PML参数
            abscissa_in_PML_half = xoriginleft - x_vals_half
            mask_half = abscissa_in_PML_half >= 0.0
            abscissa_normalized_half = cp.where(mask_half, abscissa_in_PML_half / thickness_PML_x, 0.0)
            self.d_x_half = cp.where(mask_half, d0_x * abscissa_normalized_half**self.NPOWER, self.d_x_half)
            self.K_x_half = cp.where(mask_half, 1.0 + (self.K_MAX_PML - 1.0) * abscissa_normalized_half**self.NPOWER, self.K_x_half)
            self.alpha_x_half = cp.where(mask_half, self.ALPHA_MAX_PML * (1.0 - abscissa_normalized_half), self.alpha_x_half)
        
        # 处理右边界PML
        if self.USE_PML_XMAX:
            abscissa_in_PML = x_vals - xoriginright
            mask = abscissa_in_PML >= 0.0
            abscissa_normalized = cp.where(mask, abscissa_in_PML / thickness_PML_x, 0.0)
            # 设置PML参数
            self.d_x = cp.where(mask, d0_x * abscissa_normalized**self.NPOWER, self.d_x)
            self.K_x = cp.where(mask, 1.0 + (self.K_MAX_PML - 1.0) * abscissa_normalized**self.NPOWER, self.K_x)
            self.alpha_x = cp.where(mask, self.ALPHA_MAX_PML * (1.0 - abscissa_normalized), self.alpha_x)
            
            abscissa_in_PML_half = x_vals_half - xoriginright
            mask_half = abscissa_in_PML_half >= 0.0
            abscissa_normalized_half = cp.where(mask_half, abscissa_in_PML_half / thickness_PML_x, 0.0)
            self.d_x_half = cp.where(mask_half, d0_x * abscissa_normalized_half**self.NPOWER, self.d_x_half)
            self.K_x_half = cp.where(mask_half, 1.0 + (self.K_MAX_PML - 1.0) * abscissa_normalized_half**self.NPOWER, self.K_x_half)
            self.alpha_x_half = cp.where(mask_half, self.ALPHA_MAX_PML * (1.0 - abscissa_normalized_half), self.alpha_x_half)
        
        # 计算CPML更新系数
        # 确保alpha值非负
        self.alpha_x = cp.maximum(self.alpha_x, self.ZERO)
        self.alpha_x_half = cp.maximum(self.alpha_x_half, self.ZERO)
        
        # b系数：时间步长相关的衰减因子
        self.b_x = cp.exp(-(self.d_x / self.K_x + self.alpha_x) * self.DELTAT)
        self.b_x_half = cp.exp(-(self.d_x_half / self.K_x_half + self.alpha_x_half) * self.DELTAT)
        
        # a系数：CPML递归公式中的系数
        mask = cp.abs(self.d_x) > 1.0e-6
        self.a_x = cp.where(mask,
                           self.d_x * (self.b_x - 1.0) / (self.K_x * (self.d_x + self.K_x * self.alpha_x)),
                           self.a_x)
        
        mask_half = cp.abs(self.d_x_half) > 1.0e-6
        self.a_x_half = cp.where(mask_half,
                                self.d_x_half * (self.b_x_half - 1.0) / (self.K_x_half * (self.d_x_half + self.K_x_half * self.alpha_x_half)),
                                self.a_x_half)
        
        # 在计算完所有参数后，打印数组统计信息
        print("\n=== X方向CPML数组参数统计 ===")
        print(f'd_x: min={cp.min(self.d_x)}, max={cp.max(self.d_x)}, mean={cp.mean(self.d_x)}')
        print(f'd_x_half: min={cp.min(self.d_x_half)}, max={cp.max(self.d_x_half)}, mean={cp.mean(self.d_x_half)}')
        print(f'K_x: min={cp.min(self.K_x)}, max={cp.max(self.K_x)}, mean={cp.mean(self.K_x)}')
        print(f'K_x_half: min={cp.min(self.K_x_half)}, max={cp.max(self.K_x_half)}, mean={cp.mean(self.K_x_half)}')
        print(f'alpha_x: min={cp.min(self.alpha_x)}, max={cp.max(self.alpha_x)}, mean={cp.mean(self.alpha_x)}')
        print(f'alpha_x_half: min={cp.min(self.alpha_x_half)}, max={cp.max(self.alpha_x_half)}, mean={cp.mean(self.alpha_x_half)}')
        print(f'b_x: min={cp.min(self.b_x)}, max={cp.max(self.b_x)}, mean={cp.mean(self.b_x)}')
        print(f'b_x_half: min={cp.min(self.b_x_half)}, max={cp.max(self.b_x_half)}, mean={cp.mean(self.b_x_half)}')
        print(f'a_x: min={cp.min(self.a_x)}, max={cp.max(self.a_x)}, mean={cp.mean(self.a_x)}')
        print(f'a_x_half: min={cp.min(self.a_x_half)}, max={cp.max(self.a_x_half)}, mean={cp.mean(self.a_x_half)}')

    def setup_pml_boundary_y(self):
        """设置y方向的PML边界条件"""
        # 计算准P波最大速度，用于d0计算
        quasi_cp_max = cp.maximum(cp.sqrt(self.c22/self.rho), cp.sqrt(self.c11/self.rho))
        
        # 定义PML区域的吸收层厚度
        thickness_PML_y = self.NPOINTS_PML * self.DELTAY
        
        # 设置反射系数（控制PML的吸收效果）
        Rcoef = cp.float64(0.001)
        
        # 检查NPOWER值的有效性
        if self.NPOWER < 1:
            raise ValueError('NPOWER必须大于1')
        
        # 计算衰减系数d0
        d0_y = -(self.NPOWER + 1) * quasi_cp_max * cp.log(Rcoef) / (2.0 * thickness_PML_y)
        print(f'd0_y = {d0_y}')
        
        # 设置衰减区域的边界位置
        yoriginleft = thickness_PML_y                           # 左边界位置
        yoriginright = (self.NY-1)*self.DELTAY - thickness_PML_y  # 右边界位置
        
        # 创建y方向的网格点坐标数组
        y_vals = cp.arange(self.NY, dtype=cp.float64) * self.DELTAY      # 整数网格点
        y_vals_half = y_vals + self.DELTAY/2.0                           # 半网格点
        
        # 处理左边界PML
        if self.USE_PML_YMIN:
            # 计算在PML区域内的位置
            abscissa_in_PML = yoriginleft - y_vals
            mask = abscissa_in_PML >= 0.0
            # 归一化位置（0到1之间）
            abscissa_normalized = cp.where(mask, abscissa_in_PML / thickness_PML_y, 0.0)
            
            # 设置整数网格点的PML参数
            self.d_y = cp.where(mask, d0_y * abscissa_normalized**self.NPOWER, self.d_y)           # 衰减函数
            self.K_y = cp.where(mask, 1.0 + (self.K_MAX_PML - 1.0) * abscissa_normalized**self.NPOWER, self.K_y)  # 拉伸函数
            self.alpha_y = cp.where(mask, self.ALPHA_MAX_PML * (1.0 - abscissa_normalized), self.alpha_y)  # 频率调制函数
            
            # 设置半网格点的PML参数
            abscissa_in_PML_half = yoriginleft - y_vals_half
            mask_half = abscissa_in_PML_half >= 0.0
            abscissa_normalized_half = cp.where(mask_half, abscissa_in_PML_half / thickness_PML_y, 0.0)
            self.d_y_half = cp.where(mask_half, d0_y * abscissa_normalized_half**self.NPOWER, self.d_y_half)
            self.K_y_half = cp.where(mask_half, 1.0 + (self.K_MAX_PML - 1.0) * abscissa_normalized_half**self.NPOWER, self.K_y_half)
            self.alpha_y_half = cp.where(mask_half, self.ALPHA_MAX_PML * (1.0 - abscissa_normalized_half), self.alpha_y_half)
        
        # 处理右边界PML（与左边界类似）
        if self.USE_PML_YMAX:
            abscissa_in_PML = y_vals - yoriginright
            mask = abscissa_in_PML >= 0.0
            abscissa_normalized = cp.where(mask, abscissa_in_PML / thickness_PML_y, 0.0)
            # 设置PML参数（与左边界相同的处理方式）
            self.d_y = cp.where(mask, d0_y * abscissa_normalized**self.NPOWER, self.d_y)
            self.K_y = cp.where(mask, 1.0 + (self.K_MAX_PML - 1.0) * abscissa_normalized**self.NPOWER, self.K_y)
            self.alpha_y = cp.where(mask, self.ALPHA_MAX_PML * (1.0 - abscissa_normalized), self.alpha_y)
            
            abscissa_in_PML_half = y_vals_half - yoriginright
            mask_half = abscissa_in_PML_half >= 0.0
            abscissa_normalized_half = cp.where(mask_half, abscissa_in_PML_half / thickness_PML_y, 0.0)
            self.d_y_half = cp.where(mask_half, d0_y * abscissa_normalized_half**self.NPOWER, self.d_y_half)
            self.K_y_half = cp.where(mask_half, 1.0 + (self.K_MAX_PML - 1.0) * abscissa_normalized_half**self.NPOWER, self.K_y_half)
            self.alpha_y_half = cp.where(mask_half, self.ALPHA_MAX_PML * (1.0 - abscissa_normalized_half), self.alpha_y_half)
        
        # 计算CPML更新系数
        # b系数：时间步长相关的衰减因子
        self.b_y = cp.exp(-(self.d_y / self.K_y + self.alpha_y) * self.DELTAT)
        self.b_y_half = cp.exp(-(self.d_y_half / self.K_y_half + self.alpha_y_half) * self.DELTAT)
        
        # a系数：CPML递归公式中的系数
        mask = cp.abs(self.d_y) > 1.0e-6
        self.a_y = cp.where(mask,
                           self.d_y * (self.b_y - 1.0) / (self.K_y * (self.d_y + self.K_y * self.alpha_y)),
                           self.a_y)
        
        mask_half = cp.abs(self.d_y_half) > 1.0e-6
        self.a_y_half = cp.where(mask_half,
                                self.d_y_half * (self.b_y_half - 1.0) / (self.K_y_half * (self.d_y_half + self.K_y_half * self.alpha_y_half)),
                                self.a_y_half)
        
        # 在计算完所有参数后，打印数组统计信息
        print("\n=== Y方向CPML数组参数统计 ===")
        print(f'd_y: min={cp.min(self.d_y)}, max={cp.max(self.d_y)}, mean={cp.mean(self.d_y)}')
        print(f'd_y_half: min={cp.min(self.d_y_half)}, max={cp.max(self.d_y_half)}, mean={cp.mean(self.d_y_half)}')
        print(f'K_y: min={cp.min(self.K_y)}, max={cp.max(self.K_y)}, mean={cp.mean(self.K_y)}')
        print(f'K_y_half: min={cp.min(self.K_y_half)}, max={cp.max(self.K_y_half)}, mean={cp.mean(self.K_y_half)}')
        print(f'alpha_y: min={cp.min(self.alpha_y)}, max={cp.max(self.alpha_y)}, mean={cp.mean(self.alpha_y)}')
        print(f'alpha_y_half: min={cp.min(self.alpha_y_half)}, max={cp.max(self.alpha_y_half)}, mean={cp.mean(self.alpha_y_half)}')
        print(f'b_y: min={cp.min(self.b_y)}, max={cp.max(self.b_y)}, mean={cp.mean(self.b_y)}')
        print(f'b_y_half: min={cp.min(self.b_y_half)}, max={cp.max(self.b_y_half)}, mean={cp.mean(self.b_y_half)}')
        print(f'a_y: min={cp.min(self.a_y)}, max={cp.max(self.a_y)}, mean={cp.mean(self.a_y)}')
        print(f'a_y_half: min={cp.min(self.a_y_half)}, max={cp.max(self.a_y_half)}, mean={cp.mean(self.a_y_half)}')

    def apply_boundary_conditions(self):
        """应用Dirichlet边界条件（刚性边界）"""
        # 设置x方向速度边界条件
        self.vx[0,:] = self.ZERO    # 左边界
        self.vx[-1,:] = self.ZERO   # 右边界
        self.vx[:,0] = self.ZERO    # 下边界
        self.vx[:,-1] = self.ZERO   # 上边界
        
        # 设置y方向速度边界条件
        self.vy[0,:] = self.ZERO    # 左边界
        self.vy[-1,:] = self.ZERO   # 右边界
        self.vy[:,0] = self.ZERO    # 下边界
        self.vy[:,-1] = self.ZERO   # 上边界

    def compute_wave_propagation(self):
        """在 GPU 上计算波场传播"""
        # 使用更合适的线程块大小
        block_dim = (16, 16)  # 减小线程块大小以获得更好的网格覆盖
        
        # 计算网格维度
        grid_x = (self.NX + block_dim[0] - 1) // block_dim[0]
        grid_y = (self.NY + block_dim[1] - 1) // block_dim[1]
        grid_dim = (grid_x, grid_y)
        
        # 打印网格信息用于调试
        if not hasattr(self, '_grid_info_printed'):
            total_threads_x = grid_x * block_dim[0]
            total_threads_y = grid_y * block_dim[1]
            print(f"计算网格信息:")
            print(f"数据维度: ({self.NX}, {self.NY})")
            print(f"线程块维度: {block_dim}")
            print(f"网格维度: {grid_dim}")
            print(f"总线程数: {total_threads_x} x {total_threads_y}")
            print(f"多余覆盖: X={total_threads_x-self.NX}, Y={total_threads_y-self.NY}")
            self._grid_info_printed = True

        # 计算应力场和速度场
        compute_stress(grid_dim, block_dim, (
            self.vx, self.vy,               # 速度场
            self.sigmaxx, self.sigmayy, self.sigmaxy,  # 应力场
            self.memory_dvx_dx, self.memory_dvy_dy,    # 记忆变量
            self.memory_dvy_dx, self.memory_dvx_dy,
            self.b_x_half, self.b_y,                   # PML参数
            self.a_x_half, self.a_y,
            self.b_x, self.b_y_half,
            self.a_x, self.a_y_half,
            self.K_x_half, self.K_y,                   # 拉伸系数
            self.K_x, self.K_y_half,
            self.DELTAX, self.DELTAY, self.DELTAT,     # 网格参数
            self.c11, self.c12, self.c22, self.c33,    # 材料参数
            self.NX, self.NY                           # 网格大小
        ))

        # 计算速度场
        compute_velocity(grid_dim, block_dim, (
            self.sigmaxx, self.sigmayy, self.sigmaxy,  # 应力场
            self.vx, self.vy,                          # 速度场
            self.memory_dsigmaxx_dx, self.memory_dsigmaxy_dy,  # 记忆变量
            self.memory_dsigmaxy_dx, self.memory_dsigmayy_dy,
            self.b_x, self.b_y,                        # PML参数
            self.a_x, self.a_y,
            self.b_x_half, self.b_y_half,
            self.a_x_half, self.a_y_half,
            self.K_x, self.K_y,                        # 拉伸系数
            self.K_x_half, self.K_y_half,
            self.DELTAX, self.DELTAY, self.DELTAT,     # 网格参数
            self.rho,                                  # 密度
            self.NX, self.NY                           # 网格大小
        ))


    def record_seismograms(self, it):
        """在检波器位置记录地震图数据"""
        # 将数据从GPU移动到CPU以进行记录
        vx_cpu = cp.asnumpy(self.vx)  # x方向速度场
        vy_cpu = cp.asnumpy(self.vy)  # y方向速度场
        
        # 在每个检波器位置记录速度值
        for i in range(self.NREC):  # 遍历所有检波器
            # 记录水平和垂直分量的速度
            self.seismogram_vx[it-1, i] = vx_cpu[self.rec_x[i], self.rec_z[i]]  # 记录x分量
            self.seismogram_vz[it-1, i] = vy_cpu[self.rec_x[i], self.rec_z[i]]  # 记录y分量

    def output_info(self, it):
        """输出模拟状态信息和波场快照"""
        # 将数据从GPU移动到CPU以进行可视化
        vx_cpu = cp.asnumpy(self.vx)  # x方向速度场
        vy_cpu = cp.asnumpy(self.vy)  # y方向速度场
        
        # 计算速度场的最大幅值（用于稳定性检查）
        velocnorm = np.max(np.sqrt(vx_cpu**2 + vy_cpu**2))
        
        # 输出当前时间步信息
        print(f'时间步: {it}/{self.NSTEP}')
        print(f'模拟时间: {(it-1)*self.DELTAT:.6f} 秒')
        print(f'速度矢量最大范数 (m/s) = {velocnorm}')
        print()
        
        # 检查数值稳定性
        if velocnorm > self.STABILITY_THRESHOLD:
            raise RuntimeError('模拟变得不稳定并发散')
        
        # 生成波场快照图像
        #self.create_color_image(vx_cpu, it, 1)  # 生成x方向速度场快照
        #self.create_color_image(vy_cpu, it, 2)  # 生成y方向速度场快照

    def plot_seismograms(self):
        """绘制所有检波器的地震记录"""
        # 创建图形窗口
        plt.figure(figsize=(15, 10))
        
        # 设置全局字体
        try:
            plt.rcParams['font.family'] = 'Times New Roman'
        except:
            # 如果Times New Roman不可用，使用默认字体
            plt.rcParams['font.family'] = 'serif'
        
        # 绘制水平分量地震记录
        plt.subplot(211)  # 上半部分显示水平分量
        plt.imshow(self.seismogram_vx, aspect='auto', cmap='gray',
                  extent=[0, self.NREC-1, self.NSTEP, 0])  # 使用灰度图显示
        plt.colorbar(label='Amplitude')  # 添加颜色条
        plt.title('Horizontal Component Seismogram')
        plt.xlabel('Receiver Number')
        plt.ylabel('Time Step (nt)')
        
        # 绘制垂直分量地震记录
        plt.subplot(212)  # 下半部分显示垂直分量
        plt.imshow(self.seismogram_vz, aspect='auto', cmap='gray',
                  extent=[0, self.NREC-1, self.NSTEP, 0])
        plt.colorbar(label='Amplitude')
        plt.title('Vertical Component Seismogram')
        plt.xlabel('Receiver Number')
        plt.ylabel('Time Step (nt)')
        
        # 调整布局并保存图像
        plt.tight_layout()  # 自动调整子图之间的间距
        plt.savefig(os.path.join(self.output_dir, 'seismograms.png'), dpi=300)  # 保存高分辨率图像
        plt.close()  # 关闭图形窗口释放内存

    def create_color_image(self, image_data_2D, it, field_number):
        """创建波场分量的彩色可视化图像
        
        参数:
            image_data_2D: 2D数组，包含要可视化的波场数据
            it: 整数，当前时间步
            field_number: 整数，1表示Vx分量，2表示Vy分量
        """
        # 可视化参数设置
        POWER_DISPLAY = 0.30  # 非线性显示指数，用于增强小振幅的显示效果
        cutvect = 0.01       # 振幅阈值，低于此值的点将使用背景色
        WHITE_BACKGROUND = True  # 背景色选择：True为白色，False为黑色
        width_cross = 5      # 震源标记十字的宽度(像素)
        thickness_cross = 1  # 震源标记十字的粗细(像素)
        
        # 根据field_number确定输出文件名
        if field_number == 1:
            field_name = 'Vx'  # x方向速度分量
        else:
            field_name = 'Vy'  # y方向速度分量
        
        # 生成输出文件名，格式为"imageXXXXXX_Vx/y.png"
        fig_name = f'image{it:06d}_{field_name}.png'
        
        # 直接使用self.output_dir作为输出路径
        fig_name = os.path.join(self.output_dir, fig_name)
        
        # 计算波场的最大绝对振幅，用于归一化
        max_amplitude = np.max(np.abs(image_data_2D))
        
        # 创建RGB图像数组，维度为(NY, NX, 3)
        img = np.zeros((self.NY, self.NX, 3))
        
        # 逐点填充图像数组
        for iy in range(self.NY-1, -1, -1):  # 从上到下遍历
            for ix in range(self.NX):         # 从左到右遍历
                # 将数值归一化到[-1,1]范围
                normalized_value = image_data_2D[ix,iy] / max_amplitude
                normalized_value = np.clip(normalized_value, -1.0, 1.0)  # 限制在[-1,1]范围内
                
                # 绘制震源位置（橙色十字）
                if ((ix >= self.ISOURCE - width_cross and ix <= self.ISOURCE + width_cross and 
                     iy >= self.JSOURCE - thickness_cross and iy <= self.JSOURCE + thickness_cross) or
                    (ix >= self.ISOURCE - thickness_cross and ix <= self.ISOURCE + thickness_cross and
                     iy >= self.JSOURCE - width_cross and iy <= self.JSOURCE + width_cross)):
                    img[iy,ix] = [1.0, 0.616, 0.0]  # 橙色
                
                # 绘制边框（黑色）
                elif ix <= 1 or ix >= self.NX-2 or iy <= 1 or iy >= self.NY-2:
                    img[iy,ix] = [0.0, 0.0, 0.0]  # 黑色
                
                # 绘制PML边界（橙黄色）
                elif ((self.USE_PML_XMIN and ix == self.NPOINTS_PML) or
                      (self.USE_PML_XMAX and ix == self.NX - self.NPOINTS_PML) or
                      (self.USE_PML_YMIN and iy == self.NPOINTS_PML) or
                      (self.USE_PML_YMAX and iy == self.NY - self.NPOINTS_PML)):
                    img[iy,ix] = [1.0, 0.588, 0.0]  # 橙黄色
                
                # 绘制检波器位置（绿色）
                elif any((ix == rx and iy == rz) for rx, rz in zip(self.rec_x, self.rec_z)):
                    img[iy,ix] = [0.0, 1.0, 0.0]  # 绿色
                
                # 处理低于阈值的点
                elif abs(image_data_2D[ix,iy]) <= max_amplitude * cutvect:
                    if WHITE_BACKGROUND:
                        img[iy,ix] = [1.0, 1.0, 1.0]  # 白色背景
                    else:
                        img[iy,ix] = [0.0, 0.0, 0.0]  # 黑色背景
                
                # 处理正常波场值
                else:
                    if normalized_value >= 0.0:
                        # 正值用红色表示，强度随值变化
                        img[iy,ix] = [normalized_value**POWER_DISPLAY, 0.0, 0.0]
                    else:
                        # 负值用蓝色表示，强度随值变化
                        img[iy,ix] = [0.0, 0.0, abs(normalized_value)**POWER_DISPLAY]
        
        # 保存图像为PNG文件
        plt.imsave(fig_name, img)

    def save_shot_records(self):
        """保存多炮地震记录"""
        # 当前现是单炮记录
        self.shot_records_vx[0] = self.seismogram_vx  # shape: (NSTEP, NREC)
        self.shot_records_vz[0] = self.seismogram_vz  # shape: (NSTEP, NREC)
        
        # 保存为numpy数组文件
        np.save(os.path.join(self.output_dir, 'shot_records_vx.npy'), self.shot_records_vx)
        np.save(os.path.join(self.output_dir, 'shot_records_vz.npy'), self.shot_records_vz)
        
        # 保存参数信息
        params = {
            'NSHOT': self.NSHOT,    # 炮数
            'NSTEP': self.NSTEP,    # 时间步数
            'NREC': self.NREC,      # 检波器数
            'dt': self.DELTAT,      # 时间采样间隔
            'dx': self.DELTAX,      # 空间采样间隔
            'source_positions': [(self.ISOURCE, self.JSOURCE)],  # 震源位置
            'receiver_positions': list(zip(self.rec_x, self.rec_z))  # 检波器位置
        }
        np.save(os.path.join(self.output_dir, 'simulation_params.npy'), params)

    def simulate(self):
        """运行主模拟程序"""
        # 1. 初始检查
        quasi_cp_max = cp.maximum(cp.sqrt(self.c22/self.rho), cp.sqrt(self.c11/self.rho))
        Courant_number = quasi_cp_max * self.DELTAT * cp.sqrt(1.0/self.DELTAX**2 + 1.0/self.DELTAY**2)
        print(f'Courant数为 {float(Courant_number)}')
        if Courant_number > 1.0:
            raise ValueError('时间步长过大，模拟将不稳定')
        
        # 2. 网格覆盖检查（仅在开始时执行一次）
        self.check_grid_coverage()
        
        # 3. 设置PML吸收边界
        self.setup_pml_boundary_x()
        self.setup_pml_boundary_y()
        print("\n=== 开始时间步进计算 ===")
        
        # 4. 时间步进主循环
        for it in range(1, self.NSTEP + 1):
            # 计算波场传播
            self.compute_wave_propagation()  
            
            # 添加震源
            self.add_source(it)
            
            # 应用边界条件
            self.apply_boundary_conditions()
            
            # 记录地震图
            self.record_seismograms(it)
            
            # 仅进行稳定性检查，不添加数值耗散
            self.check_stability(it)
            
            # 输出进度信息
            if it % 100 == 0:
                print(f'时间步 {it}/{self.NSTEP}...')
            
            # 输出详细信息和波场快照
            if it % self.IT_DISPLAY == 0:
                self.output_info(it)
        
        print("\n=== 模拟完成 ===")
        
        # 5. 保存结果
        self.save_shot_records()
        self.plot_seismograms()

    def check_stability(self, it):
        """检查数值稳定性"""
        max_vx = float(cp.max(cp.abs(self.vx)))
        max_vy = float(cp.max(cp.abs(self.vy)))
        
        # 记录最大值历史
        if not hasattr(self, 'max_v_history'):
            self.max_v_history = []
        self.max_v_history.append((max_vx, max_vy))
        
        # 检查突变
        if len(self.max_v_history) > 1:
            prev_max = self.max_v_history[-2]
            current_max = self.max_v_history[-1]
            change_ratio = max(current_max[0]/prev_max[0] if prev_max[0] != 0 else 1,
                             current_max[1]/prev_max[1] if prev_max[1] != 0 else 1)
            
            # 修改阈值，允许更大的变化率，特别是在早期时间步
            threshold = 5.0 if it < 10 else 2.0  # 早期允许更大的变化
            if change_ratio > threshold:
                print(f"警告：时间步{it}检测到速度场突变")
                print(f"前一步最大值: vx={prev_max[0]:.2e}, vy={prev_max[1]:.2e}")
                print(f"当前步最大值: vx={current_max[0]:.2e}, vy={current_max[1]:.2e}")
                print(f"变化率: {change_ratio:.2f}")

    def check_grid_coverage(self):
        """检查CUDA网格覆盖情况"""
        block_dim = (16, 16)
        grid_x = (self.NX + block_dim[0] - 1) // block_dim[0]
        grid_y = (self.NY + block_dim[1] - 1) // block_dim[1]
        
        total_threads_x = grid_x * block_dim[0]
        total_threads_y = grid_y * block_dim[1]
        
        print(f"网格覆盖分析:")
        print(f"X方向: 需要{self.NX}个点, 实际覆盖{total_threads_x}个点")
        print(f"Y方向: 需要{self.NY}个点, 实际覆盖{total_threads_y}个点")
        print(f"多余覆盖: X={total_threads_x-self.NX}, Y={total_threads_y-self.NY}")

if __name__ == '__main__':
    simulator = SeismicCPML2DAniso()
    simulator.simulate()