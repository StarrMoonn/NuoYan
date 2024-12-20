import numpy as np
import matplotlib.pyplot as plt
import os
import cupy as cp

class SeismicCPML2DAniso:
    def __init__(self):
        # Grid parameters
        self.NX = 401
        self.NY = 401
        
        # Grid spacing
        self.DELTAX = 0.0625e-2
        self.DELTAY = self.DELTAX
        
        # PML flags and parameters  
        self.USE_PML_XMIN = True
        self.USE_PML_XMAX = True
        self.USE_PML_YMIN = True
        self.USE_PML_YMAX = True
        self.NPOINTS_PML = 10
        
        # Material properties (Model I from Becache, Fauqueux and Joly)
        self.scale_aniso = 1.0e10
        self.c11 = 4.0 * self.scale_aniso
        self.c12 = 3.8 * self.scale_aniso  
        self.c22 = 20.0 * self.scale_aniso
        self.c33 = 2.0 * self.scale_aniso
        self.rho = 4000.0
        self.f0 = 200.0e3
        
        # Time stepping parameters
        self.NSTEP = 3000  # Changed to match Fortran version
        self.DELTAT = 50.0e-9
        
        # Source parameters
        self.t0 = 1.20/self.f0
        self.factor = 1.0e7
        self.ISOURCE = self.NX // 2
        self.JSOURCE = self.NY // 2
        self.xsource = (self.ISOURCE - 1) * self.DELTAX
        self.ysource = (self.JSOURCE - 1) * self.DELTAY
        self.ANGLE_FORCE = 0.0
        
        # Receiver parameters
        self.NREC = 50  # Number of receivers
        self.first_rec_x = 100  # First receiver x position (grid points)
        self.first_rec_z = 50   # First receiver z position (grid points)
        self.rec_dx = 4         # Receiver x spacing (grid points)
        self.rec_dz = 0         # Receiver z spacing (grid points)
        
        # Initialize receiver arrays
        self.rec_x = np.zeros(self.NREC, dtype=np.int32)
        self.rec_z = np.zeros(self.NREC, dtype=np.int32)
        
        # Initialize seismogram arrays
        self.seismogram_vx = np.zeros((self.NSTEP, self.NREC))
        self.seismogram_vz = np.zeros((self.NSTEP, self.NREC))
        
        # Display parameters
        self.IT_DISPLAY = 100
        
        # Constants
        self.PI = cp.pi
        self.DEGREES_TO_RADIANS = self.PI / 180.0
        self.ZERO = cp.float64(0.0)
        self.HUGEVAL = cp.float64(1.0e+30)
        self.STABILITY_THRESHOLD = cp.float64(1.0e+25)
        
        # PML parameters
        self.NPOWER = cp.float64(2.0)
        self.K_MAX_PML = cp.float64(1.0)
        self.ALPHA_MAX_PML = cp.float64(2.0 * self.PI * (self.f0/2.0))
        
        # Initialize arrays
        self.initialize_arrays()
        
        # Setup receivers
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

    def setup_receivers(self):
        """Setup receiver positions"""
        for i in range(self.NREC):
            self.rec_x[i] = self.first_rec_x + i * self.rec_dx
            self.rec_z[i] = self.first_rec_z + i * self.rec_dz
            
        # Check if receivers are within grid bounds
        if np.any(self.rec_x >= self.NX) or np.any(self.rec_z >= self.NY):
            raise ValueError("Receiver positions exceed grid dimensions")

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

    def setup_pml_boundary_y(self):
        """设置y方向的PML边界条件"""
        # 计算准P波最大速度，用于d0计算
        quasi_cp_max = cp.maximum(cp.sqrt(self.c22/self.rho), cp.sqrt(self.c11/self.rho))
        
        # 定义PML区域的吸收层厚度
        thickness_PML_y = self.NPOINTS_PML * self.DELTAY
        
        # 设置反射系数（控制PML的吸收效果）
        Rcoef = cp.float64(0.001)
        
        # 计算衰减系数d0
        d0_y = -(self.NPOWER + 1) * quasi_cp_max * cp.log(Rcoef) / (2.0 * thickness_PML_y)
        
        # 检查NPOWER值的有效性
        if self.NPOWER < 1:
            raise ValueError('NPOWER必须大于1')
        
        # 计算 d0
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

    def compute_stress(self):
        """计算应力分量并更新记忆变量"""
        # 计算速度梯度
        value_dvx_dx = (self.vx[1:,:] - self.vx[:-1,:]) / self.DELTAX    # x方向速度对x的导数
        value_dvy_dy = cp.zeros_like(value_dvx_dx)
        value_dvy_dy[:,1:] = (self.vy[:-1,1:] - self.vy[:-1,:-1]) / self.DELTAY  # y方向速度对y的导数
        
        # 更新PML记忆变量
        self.memory_dvx_dx[:-1,:] = (self.b_x_half[:-1,None] * self.memory_dvx_dx[:-1,:] + 
                                    self.a_x_half[:-1,None] * value_dvx_dx)
        self.memory_dvy_dy[:-1,1:] = (self.b_y[1:,None].T * self.memory_dvy_dy[:-1,1:] + 
                                     self.a_y[1:,None].T * value_dvy_dy[:,1:])
        
        # 将PML效应加入到导数计算中
        value_dvx_dx = value_dvx_dx / self.K_x_half[:-1,None] + self.memory_dvx_dx[:-1,:]
        value_dvy_dy[:,1:] = value_dvy_dy[:,1:] / self.K_y[1:,None].T + self.memory_dvy_dy[:-1,1:]
        
        # 更新正应力分量
        self.sigmaxx[:-1,1:] += (self.c11 * value_dvx_dx[:,1:] + self.c12 * value_dvy_dy[:,1:]) * self.DELTAT
        self.sigmayy[:-1,1:] += (self.c12 * value_dvx_dx[:,1:] + self.c22 * value_dvy_dy[:,1:]) * self.DELTAT

        # 计算剪应力部分
        # 使用向量化操作计算速度梯度
        value_dvy_dx = (self.vy[1:,:-1] - self.vy[:-1,:-1]) / self.DELTAX    # vy对x的偏导数
        value_dvx_dy = (self.vx[1:,1:] - self.vx[1:,:-1]) / self.DELTAY     # vx对y的偏导数
        
        # 更新PML记忆变量
        # 对x方向的记忆变量进行更新
        self.memory_dvy_dx[1:,:-1] = (self.b_x[1:,None] * self.memory_dvy_dx[1:,:-1] + 
                                     self.a_x[1:,None] * value_dvy_dx)
        # 对y方向的记忆变量进行更新
        self.memory_dvx_dy[1:,:-1] = (self.b_y_half[:-1,None].T * self.memory_dvx_dy[1:,:-1] + 
                                     self.a_y_half[:-1,None].T * value_dvx_dy)
        
        # 将PML效应加入到导数计算中
        value_dvy_dx = value_dvy_dx / self.K_x[1:,None] + self.memory_dvy_dx[1:,:-1]      # x方向PML修正
        value_dvx_dy = value_dvx_dy / self.K_y_half[:-1,None].T + self.memory_dvx_dy[1:,:-1]  # y方向PML修正
        
        # 更新剪应力分量
        self.sigmaxy[1:,:-1] += self.c33 * (value_dvy_dx + value_dvx_dy) * self.DELTAT

    def compute_velocity(self):
        """计算速度分量并更新记忆变量"""
        # 计算x方向速度所需的应力梯度
        value_dsigmaxx_dx = (self.sigmaxx[1:,1:] - self.sigmaxx[:-1,1:]) / self.DELTAX   # σxx对x的偏导数
        value_dsigmaxy_dy = (self.sigmaxy[1:,1:] - self.sigmaxy[1:,:-1]) / self.DELTAY   # σxy对y的偏导数
        
        # 更新x方向速度相关的PML记忆变量
        self.memory_dsigmaxx_dx[1:,1:] = (self.b_x[1:,None] * self.memory_dsigmaxx_dx[1:,1:] + 
                                         self.a_x[1:,None] * value_dsigmaxx_dx)
        self.memory_dsigmaxy_dy[1:,1:] = (self.b_y[1:,None].T * self.memory_dsigmaxy_dy[1:,1:] + 
                                         self.a_y[1:,None].T * value_dsigmaxy_dy)
        
        # 将PML效应加入到应力梯度计算中
        value_dsigmaxx_dx = value_dsigmaxx_dx / self.K_x[1:,None] + self.memory_dsigmaxx_dx[1:,1:]
        value_dsigmaxy_dy = value_dsigmaxy_dy / self.K_y[1:,None].T + self.memory_dsigmaxy_dy[1:,1:]
        
        # 更新x方向速度分量
        self.vx[1:,1:] += (value_dsigmaxx_dx + value_dsigmaxy_dy) * self.DELTAT / self.rho
        
        # 计算y方向速度所需的应力梯度
        value_dsigmaxy_dx = (self.sigmaxy[1:,:-1] - self.sigmaxy[:-1,:-1]) / self.DELTAX  # σxy对x的偏导数
        value_dsigmayy_dy = (self.sigmayy[:-1,1:] - self.sigmayy[:-1,:-1]) / self.DELTAY  # σyy对y的偏导数
        
        # 更新y方向速度相关的PML记忆变量
        self.memory_dsigmaxy_dx[:-1,:-1] = (self.b_x_half[:-1,None] * self.memory_dsigmaxy_dx[:-1,:-1] + 
                                           self.a_x_half[:-1,None] * value_dsigmaxy_dx)
        self.memory_dsigmayy_dy[:-1,:-1] = (self.b_y_half[:-1,None].T * self.memory_dsigmayy_dy[:-1,:-1] + 
                                           self.a_y_half[:-1,None].T * value_dsigmayy_dy)
        
        # 将PML效应加入到应力梯度计算中
        value_dsigmaxy_dx = value_dsigmaxy_dx / self.K_x_half[:-1,None] + self.memory_dsigmaxy_dx[:-1,:-1]
        value_dsigmayy_dy = value_dsigmayy_dy / self.K_y_half[:-1,None].T + self.memory_dsigmayy_dy[:-1,:-1]
        
        # 更新y方向速度分量
        self.vy[:-1,:-1] += (value_dsigmaxy_dx + value_dsigmayy_dy) * self.DELTAT / self.rho

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
        self.create_color_image(vx_cpu, it, 1)  # 生成x方向速度场快照
        self.create_color_image(vy_cpu, it, 2)  # 生成y方向速度场快照

    def plot_seismograms(self):
        """绘制所有检波器的地震记录"""
        # 创建图形窗口
        plt.figure(figsize=(15, 10))
        
        # 绘制水平分量地震记录
        plt.subplot(211)  # 上半部分显示水平分量
        plt.imshow(self.seismogram_vx, aspect='auto', cmap='gray',
                  extent=[0, self.NREC-1, self.NSTEP, 0])  # 使用灰度图显示
        plt.colorbar(label='振幅')  # 添加颜色条
        plt.title('水平分量地震记录')
        plt.xlabel('检波器编号')
        plt.ylabel('时间步 (nt)')
        
        # 绘制垂直分量地震记录
        plt.subplot(212)  # 下半部分显示垂直分量
        plt.imshow(self.seismogram_vz, aspect='auto', cmap='gray',
                  extent=[0, self.NREC-1, self.NSTEP, 0])
        plt.colorbar(label='振幅')
        plt.title('垂直分量地震记录')
        plt.xlabel('检波器编号')
        plt.ylabel('时间步 (nt)')
        
        # 调整布局并保存图像
        plt.tight_layout()  # 自动调整子图之间的间距
        plt.savefig('seismograms.png')  # 保存为PNG文件
        plt.close()  # 关闭图形窗口释放内存

    def create_color_image(self, image_data_2D, it, field_number):
        """Create a color image of a given vector component"""
        # Parameters for visualization
        POWER_DISPLAY = 0.30  # non linear display to enhance small amplitudes
        cutvect = 0.01       # amplitude threshold above which we draw the color point
        WHITE_BACKGROUND = True
        width_cross = 5      # size of cross and square in pixels
        thickness_cross = 1
        
        # Create figure name based on field number
        if field_number == 1:
            field_name = 'Vx'
        else:
            field_name = 'Vy'
        
        fig_name = f'image{it:06d}_{field_name}.png'
        
        # Compute maximum amplitude
        max_amplitude = np.max(np.abs(image_data_2D))
        
        # Create RGB array for image
        img = np.zeros((self.NY, self.NX, 3))
        
        # Fill the image array
        for iy in range(self.NY-1, -1, -1):
            for ix in range(self.NX):
                # Normalize value to [-1,1]
                normalized_value = image_data_2D[ix,iy] / max_amplitude
                
                # Clip values to [-1,1]
                normalized_value = np.clip(normalized_value, -1.0, 1.0)
                
                # Draw source location
                if ((ix >= self.ISOURCE - width_cross and ix <= self.ISOURCE + width_cross and 
                     iy >= self.JSOURCE - thickness_cross and iy <= self.JSOURCE + thickness_cross) or
                    (ix >= self.ISOURCE - thickness_cross and ix <= self.ISOURCE + thickness_cross and
                     iy >= self.JSOURCE - width_cross and iy <= self.JSOURCE + width_cross)):
                    img[iy,ix] = [1.0, 0.616, 0.0]  # Orange
                
                # Draw frame
                elif ix <= 1 or ix >= self.NX-2 or iy <= 1 or iy >= self.NY-2:
                    img[iy,ix] = [0.0, 0.0, 0.0]  # Black
                
                # Draw PML boundaries
                elif ((self.USE_PML_XMIN and ix == self.NPOINTS_PML) or
                      (self.USE_PML_XMAX and ix == self.NX - self.NPOINTS_PML) or
                      (self.USE_PML_YMIN and iy == self.NPOINTS_PML) or
                      (self.USE_PML_YMAX and iy == self.NY - self.NPOINTS_PML)):
                    img[iy,ix] = [1.0, 0.588, 0.0]  # Orange-yellow
                
                # Draw receivers
                elif any((ix == rx and iy == rz) for rx, rz in zip(self.rec_x, self.rec_z)):
                    img[iy,ix] = [0.0, 1.0, 0.0]  # Green
                
                # Values below threshold
                elif abs(image_data_2D[ix,iy]) <= max_amplitude * cutvect:
                    if WHITE_BACKGROUND:
                        img[iy,ix] = [1.0, 1.0, 1.0]  # White
                    else:
                        img[iy,ix] = [0.0, 0.0, 0.0]  # Black
                
                # Regular points
                else:
                    if normalized_value >= 0.0:
                        # Red for positive values
                        img[iy,ix] = [normalized_value**POWER_DISPLAY, 0.0, 0.0]
                    else:
                        # Blue for negative values
                        img[iy,ix] = [0.0, 0.0, abs(normalized_value)**POWER_DISPLAY]
        
        # Save the image
        plt.imsave(fig_name, img)

    def save_shot_records(self):
        """保存多炮地震记录"""
        # 当前实现是单炮记录
        self.shot_records_vx[0] = self.seismogram_vx  # shape: (NSTEP, NREC)
        self.shot_records_vz[0] = self.seismogram_vz  # shape: (NSTEP, NREC)
        
        # 保存为numpy数组文件
        np.save('shot_records_vx.npy', self.shot_records_vx)
        np.save('shot_records_vz.npy', self.shot_records_vz)
        
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
        np.save('simulation_params.npy', params)

    def simulate(self):
        """Run the main simulation"""
        # Check Courant stability condition
        quasi_cp_max = cp.maximum(cp.sqrt(self.c22/self.rho), cp.sqrt(self.c11/self.rho))
        Courant_number = quasi_cp_max * self.DELTAT * cp.sqrt(1.0/self.DELTAX**2 + 1.0/self.DELTAY**2)
        print(f'Courant number is {float(Courant_number)}')
        if Courant_number > 1.0:
            raise ValueError('Time step is too large, simulation will be unstable')
        
        # Setup PML boundaries
        self.setup_pml_boundary()
        self.setup_pml_boundary_y()
        
        # Time stepping
        for it in range(1, self.NSTEP + 1):
            if it % 100 == 0:
                print(f'Processing step {it}/{self.NSTEP}...')
            
            # Compute stress sigma and update memory variables
            self.compute_stress()
            
            # Compute velocity and update memory variables
            self.compute_velocity()
            
            # Add source
            self.add_source(it)
            
            # Apply Dirichlet boundary conditions
            self.apply_boundary_conditions()
            
            # Record seismograms
            self.record_seismograms(it)
            
            # Output information
            if it % self.IT_DISPLAY == 0 or it == 5:
                self.output_info(it)
        
        # 保存地震记录
        self.save_shot_records()
        
        # 绘制地震记录
        self.plot_seismograms()
        print("\nEnd of the simulation")

if __name__ == '__main__':
    simulator = SeismicCPML2DAniso()
    simulator.simulate()