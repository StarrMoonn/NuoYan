import numpy as np
import matplotlib.pyplot as plt
import os
import cupy as cp

class SeismicCPML2DAniso:
    def __init__(self):
        # 网格参数
        self.NX = 401  # 水平方向网格点数
        self.NY = 401  # 垂直方向网格点数
        
        # 网格间距
        self.DELTAX = 0.0625e-2 # 水平方向网格间距(m)
        self.DELTAY = 0.0625e-2  # 垂直方向网格间距(m)
        
        # PML边界参数设置
        self.USE_PML_XMIN = True  # 是否使用左边界PML层
        self.USE_PML_XMAX = True  # 是否使用右边界PML层
        self.USE_PML_YMIN = True  # 是否使用上边界PML层
        self.USE_PML_YMAX = True  # 是否使用下边界PML层
        self.NPOINTS_PML = 10     # PML层的厚度(网格点数)
        
        # 材料属性参数 (来自Becache, Fauqueux和Joly的模型I)
        self.c11 = 4.0e10   # 刚度系数(Pa)
        self.c13 = 3.8e10   # 刚度系数(Pa)
        self.c33 = 20.0e10  # 刚度系数(Pa)
        self.c44 = 2.0e10   # 刚度系数(Pa)
        self.rho = 4000.0  # 密度(kg/m³)
        self.f0 = 200.0e3  # 震源主频(Hz)
        
        # 时间步进参数
        self.NSTEP = 1000      # 总时间步数
        self.DELTAT = 50.e-9 # 时间步长(s)
        
        # Source parameters
        self.t0 = 1.20/self.f0
        self.factor = 1.0e7
        self.ISOURCE = self.NX // 2
        self.JSOURCE = self.NY // 2
        self.xsource = (self.ISOURCE - 1) * self.DELTAX
        self.ysource = (self.JSOURCE - 1) * self.DELTAY
        self.ANGLE_FORCE = 0.0
        
     
        # 检波器参数
        self.NREC = 50          # 检波器数量
        self.first_rec_x = 100  # 第一个检波器x方向位置(网格点)
        self.first_rec_z = 50   # 第一个检波器z方向位置(网格点)
        self.rec_dx = 4         # 检波器x方向间距(网格点)
        self.rec_dz = 0         # 检波器z方向间距(网格点)
        
        # 初始化检波器数组
        self.rec_x = np.zeros(self.NREC, dtype=np.int32)  # 检波器x坐标数组
        self.rec_z = np.zeros(self.NREC, dtype=np.int32)  # 检波器z坐标数组
        
        # 初始化地震记录数组
        self.seismogram_vx = np.zeros((self.NSTEP, self.NREC))  # x方向速度记录
        self.seismogram_vz = np.zeros((self.NSTEP, self.NREC))  # z方向速度记录
        
        # 显示参数
        self.IT_DISPLAY = 100  # 每隔多少步显示一次结果
        
        # 常量定义
        self.PI = cp.pi  # 圆周率
        self.DEGREES_TO_RADIANS = self.PI / 180.0  # 角度转弧度系数
        self.ZERO = cp.float64(0.0)  # 零值常量
        self.HUGEVAL = cp.float64(1.0e+30)  # 极大值常量
        self.STABILITY_THRESHOLD = cp.float64(1.0e+25)  # 稳定性阈值
        
        # PML参数
        self.NPOWER = cp.float64(2.0)  # PML衰减函数��幂次
        self.K_MAX_PML = cp.float64(1.0)  # PML最大衰减系数
        self.ALPHA_MAX_PML = cp.float64(2.0 * self.PI * (self.f0/2.0))  # PML最大吸收系数
        
        # 初始化数组
        self.initialize_arrays()
        
        # 设置检波器
        self.setup_receivers() 
        
        # 定义多炮记录数组
        self.NSHOT = 1  # 炮数(可以修改)
        
        # 创建两个三维数组来存储多炮记录
        # 维度: (炮点数, 时间步数, 检波器数)
        self.shot_records_vx = np.zeros((self.NSHOT, self.NSTEP, self.NREC))  # x方向速度多炮记录
        self.shot_records_vz = np.zeros((self.NSHOT, self.NSTEP, self.NREC))  # z方向速度多炮记录
        
        # 数组维度说明:
        # 第一维 (NSHOT): 不同的炮点位置
        # 第二维 (NSTEP): 时间采样点
        # 第三维 (NREC):  检波器通道数
        
        # 添加输出目录设置
        self.output_dir = "output"  # 输出目录路径
        # 创建输出目录(如果不存在)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


    def initialize_arrays(self):
        """在GPU上初始化所有需要的数组"""
        # 主要场变量数组
        # vx,vy为速度分量,sigma为应力分量
        self.vx = cp.zeros((self.NX, self.NY), dtype=cp.float64)
        self.vy = cp.zeros((self.NX, self.NY), dtype=cp.float64)
        self.sigmaxx = cp.zeros((self.NX, self.NY), dtype=cp.float64)
        self.sigmayy = cp.zeros((self.NX, self.NY), dtype=cp.float64)
        self.sigmaxy = cp.zeros((self.NX, self.NY), dtype=cp.float64)
        
        # PML边界区域的记忆变量
        # 用于存储PML边界处的导数值
        self.memory_dvx_dx = cp.zeros((self.NX, self.NY), dtype=cp.float64)
        self.memory_dvx_dy = cp.zeros((self.NX, self.NY), dtype=cp.float64)
        self.memory_dvy_dx = cp.zeros((self.NX, self.NY), dtype=cp.float64)
        self.memory_dvy_dy = cp.zeros((self.NX, self.NY), dtype=cp.float64)
        self.memory_dsigmaxx_dx = cp.zeros((self.NX, self.NY), dtype=cp.float64)
        self.memory_dsigmayy_dy = cp.zeros((self.NX, self.NY), dtype=cp.float64)
        self.memory_dsigmaxy_dx = cp.zeros((self.NX, self.NY), dtype=cp.float64)
        self.memory_dsigmaxy_dy = cp.zeros((self.NX, self.NY), dtype=cp.float64)
        
        # PML衰减系数的一维数组
        # 包含了x方向和y方向的衰减参数
        # d: 衰减函数
        # K: 拉伸坐标系数 
        # alpha: 频率依赖衰减系数
        # a,b: 递归更新系数
        self.d_x = cp.zeros(self.NX, dtype=cp.float64)
        self.d_x_half = cp.zeros(self.NX, dtype=cp.float64)
        self.K_x = cp.ones(self.NX, dtype=cp.float64)
        self.K_x_half = cp.ones(self.NX, dtype=cp.float64)
        self.alpha_x = cp.zeros(self.NX, dtype=cp.float64)
        self.alpha_x_half = cp.zeros(self.NX, dtype=cp.float64)
        self.a_x = cp.zeros(self.NX, dtype=cp.float64)
        self.a_x_half = cp.zeros(self.NX, dtype=cp.float64)
        self.b_x = cp.zeros(self.NX, dtype=cp.float64)
        self.b_x_half = cp.zeros(self.NX, dtype=cp.float64)
        
        # y方向的PML参数数组
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
        """设置PML边界条件并检查各向异性材料的稳定性"""
        
        # 检查各向异性材料PML模型的稳定性条件1
        # 根据Becache等人2003年的研究，计算第一个稳定性判据
        # 该判据涉及材料的刚度系数c11、c13、c44和c44之间的关系
        aniso_stability_criterion = ((self.c13 + self.c44)**2 - self.c11*(self.c33-self.c44)) * \
                               ((self.c13 + self.c44)**2 + self.c44*(self.c33-self.c44))
        print(f'PML各向异性稳定性判据1 (Becache et al. 2003) = {aniso_stability_criterion}')
        if aniso_stability_criterion > 0.0 and (self.USE_PML_XMIN or self.USE_PML_XMAX or 
                                          self.USE_PML_YMIN or self.USE_PML_YMAX):
            print('警告：当前各向异性材料在条件1下PML模型在数学上本质不稳定')
        
        # 检查稳定性条件2
        # 计算第二个稳定性判据，考虑c44的双倍影响
        aniso2 = (self.c13 + 2*self.c44)**2 - self.c11*self.c33
        print(f'PML���向异性稳定性判据2 (Becache et al. 2003) = {aniso2}')
        if aniso2 > 0.0 and (self.USE_PML_XMIN or self.USE_PML_XMAX or 
                             self.USE_PML_YMIN or self.USE_PML_YMAX):
            print('警告：当前各向异性材料在条件2下PML模型在数学上本质不稳定')
        
        # 检查稳定性条件3
        # 计算第三个稳定性判据，包含c44的平方项
        aniso3 = (self.c13 + self.c44)**2 - self.c11*self.c33 - self.c44**2
        print(f'PML各向异性稳定性判据3 (Becache et al. 2003) = {aniso3}')
        if aniso3 > 0.0 and (self.USE_PML_XMIN or self.USE_PML_XMAX or 
                             self.USE_PML_YMIN or self.USE_PML_YMAX):
            print('警告：当前各向异性材料在条件3下PML模型在数学上本质不稳定')

    
    def compute_stress(self):
        """计算应力张量分量并更新记忆变量"""
        # 计算内部点的应力分量，使用向量化操作
        
        # 计算x方向速度的空间导数 (中心差分格式)
        value_dvx_dx = (self.vx[1:,:] - self.vx[:-1,:]) / self.DELTAX
        
        # 初始化y方向速度的空间导数数组
        value_dvy_dy = cp.zeros_like(value_dvx_dx)
        # 计算y方向速度的空间导数 (中心差分格式)
        value_dvy_dy[:,1:] = (self.vy[:-1,1:] - self.vy[:-1,:-1]) / self.DELTAY
        
        # 更新PML区域的记忆变量
        # b_x_half和a_x_half是PML衰减系数
        # None的使用是为了广播操作，使一维数组变为二维
        self.memory_dvx_dx[:-1,:] = (self.b_x_half[:-1,None] * self.memory_dvx_dx[:-1,:] + 
                                    self.a_x_half[:-1,None] * value_dvx_dx)
        self.memory_dvy_dy[:-1,1:] = (self.b_y[1:,None].T * self.memory_dvy_dy[:-1,1:] + 
                                     self.a_y[1:,None].T * value_dvy_dy[:,1:])
        
        # 应用PML坐标拉伸和记忆变量修正
        # K_x_half和K_y是坐标拉伸系数
        value_dvx_dx = value_dvx_dx / self.K_x_half[:-1,None] + self.memory_dvx_dx[:-1,:]
        value_dvy_dy[:,1:] = value_dvy_dy[:,1:] / self.K_y[1:,None].T + self.memory_dvy_dy[:-1,1:]
        
        # 更新正应力分量
        # c11, c13, c22是各向异性弹性常数
        self.sigmaxx[:-1,1:] += (self.c11 * value_dvx_dx[:,1:] + self.c13 * value_dvy_dy[:,1:]) * self.DELTAT
        self.sigmayy[:-1,1:] += (self.c13 * value_dvx_dx[:,1:] + self.c33 * value_dvy_dy[:,1:]) * self.DELTAT

        # 计算剪切应力相关的空间导数
        value_dvy_dx = (self.vy[1:,:-1] - self.vy[:-1,:-1]) / self.DELTAX
        value_dvx_dy = (self.vx[1:,1:] - self.vx[1:,:-1]) / self.DELTAY
        
        # 更新剪切应力相关的PML记忆变量
        self.memory_dvy_dx[1:,:-1] = (self.b_x[1:,None] * self.memory_dvy_dx[1:,:-1] + 
                                     self.a_x[1:,None] * value_dvy_dx)
        self.memory_dvx_dy[1:,:-1] = (self.b_y_half[:-1,None].T * self.memory_dvx_dy[1:,:-1] + 
                                     self.a_y_half[:-1,None].T * value_dvx_dy)
        
        # 应用剪切应力的PML修正
        value_dvy_dx = value_dvy_dx / self.K_x[1:,None] + self.memory_dvy_dx[1:,:-1]
        value_dvx_dy = value_dvx_dy / self.K_y_half[:-1,None].T + self.memory_dvx_dy[1:,:-1]
        
        # 更新剪切应力分量
        # c33是剪切模量
        self.sigmaxy[1:,:-1] += self.c44 * (value_dvy_dx + value_dvx_dy) * self.DELTAT

    def compute_velocity(self):
        """计算速度分量并更新记忆变量
        
        该方法实现了弹性波动方程中速度分量的更新，包括:
        1. 计算应力张量的空间导数
        2. 更新PML区域的记忆变量
        3. 应用PML修正
        4. 更新速度分量
        """
        # 计算x方向速度所需的应力梯度
        # 使用中心差分计算sigmaxx在x方向的导数
        value_dsigmaxx_dx = (self.sigmaxx[1:,1:] - self.sigmaxx[:-1,1:]) / self.DELTAX
        # 计算sigmaxy在y方向���导数
        value_dsigmaxy_dy = (self.sigmaxy[1:,1:] - self.sigmaxy[1:,:-1]) / self.DELTAY
        
        # 更新PML区域的记忆变量
        # b_x和a_x是PML衰减系数，None的使用是为了进行维度广播
        self.memory_dsigmaxx_dx[1:,1:] = (self.b_x[1:,None] * self.memory_dsigmaxx_dx[1:,1:] + 
                                         self.a_x[1:,None] * value_dsigmaxx_dx)
        self.memory_dsigmaxy_dy[1:,1:] = (self.b_y[1:,None].T * self.memory_dsigmaxy_dy[1:,1:] + 
                                         self.a_y[1:,None].T * value_dsigmaxy_dy)
        
        # 应用PML坐标拉伸和记忆变量修正
        # K_x和K_y是坐标拉伸系数
        value_dsigmaxx_dx = value_dsigmaxx_dx / self.K_x[1:,None] + self.memory_dsigmaxx_dx[1:,1:]
        value_dsigmaxy_dy = value_dsigmaxy_dy / self.K_y[1:,None].T + self.memory_dsigmaxy_dy[1:,1:]
        
        # 更新x方向的速度分量
        # 根据运动方程: ρ∂v/∂t = ∂σ/∂x
        self.vx[1:,1:] += (value_dsigmaxx_dx + value_dsigmaxy_dy) * self.DELTAT / self.rho
        
        # 计算y方向速度所需的应力梯度
        # 计算sigmaxy在x方向的导数
        value_dsigmaxy_dx = (self.sigmaxy[1:,:-1] - self.sigmaxy[:-1,:-1]) / self.DELTAX
        # 计算sigmayy在y方向的导数
        value_dsigmayy_dy = (self.sigmayy[:-1,1:] - self.sigmayy[:-1,:-1]) / self.DELTAY
        
        # 更新y方向PML记忆变量
        # 使用half网格点的衰减系数
        self.memory_dsigmaxy_dx[:-1,:-1] = (self.b_x_half[:-1,None] * self.memory_dsigmaxy_dx[:-1,:-1] + 
                                           self.a_x_half[:-1,None] * value_dsigmaxy_dx)
        self.memory_dsigmayy_dy[:-1,:-1] = (self.b_y_half[:-1,None].T * self.memory_dsigmayy_dy[:-1,:-1] + 
                                           self.a_y_half[:-1,None].T * value_dsigmayy_dy)
        
        # 应用y方向PML修正
        value_dsigmaxy_dx = value_dsigmaxy_dx / self.K_x_half[:-1,None] + self.memory_dsigmaxy_dx[:-1,:-1]
        value_dsigmayy_dy = value_dsigmayy_dy / self.K_y_half[:-1,None].T + self.memory_dsigmayy_dy[:-1,:-1]
        
        # 更新y方向的速度分量
        self.vy[:-1,:-1] += (value_dsigmaxy_dx + value_dsigmayy_dy) * self.DELTAT / self.rho

    
    def add_source(self, it):
        
        # 计算高斯函数参数
        a = self.PI * self.PI * self.f0 * self.f0
        t = (it-1) * self.DELTAT
        
        # 计算震源项
        source_term = -self.factor * 2.0 * a * (t-self.t0) * cp.exp(-a*(t-self.t0)**2)
        
        # 计算力的分量
        force_x = cp.sin(self.ANGLE_FORCE * self.DEGREES_TO_RADIANS) * source_term
        force_y = cp.cos(self.ANGLE_FORCE * self.DEGREES_TO_RADIANS) * source_term
        
        # Define location of the source
        i = self.ISOURCE
        j = self.JSOURCE
        
        self.vx[i,j] += force_x * self.DELTAT / self.rho
        self.vy[i,j] += force_y * self.DELTAT / self.rho

    def record_seismograms(self, it):
        """记录检波器位置处的地震图数据"""
        # 将速度场数据从GPU转移到CPU
        vx_cpu = cp.asnumpy(self.vx)
        vy_cpu = cp.asnumpy(self.vy)
        
        # 直接记录到shot_records数组中，不使用中间数组
        for i in range(self.NREC):
            self.seismogram_vx[it-1, i] = vx_cpu[self.rec_x[i], self.rec_z[i]]
            self.seismogram_vz[it-1, i] = vy_cpu[self.rec_x[i], self.rec_z[i]]

    def apply_boundary_conditions(self):
        """应用边界条件（刚性边界）
        
        实现说明:
        1. 在计算域的四个边界上设置速度为零
        2. 模拟刚性边界条件，即波在边界处完全反射
        3. 分别处理x和y方向的速度分量
        """
        # 设置x方向速度分量的边界条件
        self.vx[0,:] = self.ZERO    # 左边界
        self.vx[-1,:] = self.ZERO   # 右边界
        self.vx[:,0] = self.ZERO    # 上边界
        self.vx[:,-1] = self.ZERO   # 下边界
        
        # 设置y方向速度分量的边界条件
        self.vy[0,:] = self.ZERO    # 左边界
        self.vy[-1,:] = self.ZERO   # 右边界
        self.vy[:,0] = self.ZERO    # 上边界
        self.vy[:,-1] = self.ZERO   # 下边界

    def output_info(self, it):
        """输出模拟状态信息并生成波场快照图像
        
        参数:
            it: int, 当前时间步
            
        功能:
        1. 计算并显示当前模拟状态的关键信息
        2. 检查模拟的稳定性
        3. 生成波场分量的可视化图像
        """
        # 将GPU上的速度场数据转移到CPU用于可视化
        vx_cpu = cp.asnumpy(self.vx)  # x方向速度分量
        vy_cpu = cp.asnumpy(self.vy)  # y方向速度分量
        
        # 计算速度场的最大范数（欧几里得范数）
        velocnorm = np.max(np.sqrt(vx_cpu**2 + vy_cpu**2))
        
        # 打印当前模拟状态信息
        print(f'Time step # {it} out of {self.NSTEP}')  # 显示当前时间步
        print(f'Time: {(it-1)*self.DELTAT:.6f} seconds')  # 显示实际模拟时间
        print(f'Max norm velocity vector V (m/s) = {velocnorm}')  # 显示最大速度范数
        print()
        
        # 检查模拟稳定性：如果速度超过阈值，说明计算发散
        if velocnorm > self.STABILITY_THRESHOLD:
            raise RuntimeError('code became unstable and blew up')
        
        self.create_color_image(vx_cpu, it, 1)
        self.create_color_image(vy_cpu, it, 2)

    def plot_seismograms(self):
        """Plot seismograms for all receivers"""
        time = np.arange(self.NSTEP) * self.DELTAT
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot horizontal component seismogram
        plt.subplot(211)
        plt.imshow(self.seismogram_vx, aspect='auto', cmap='seismic',
                  extent=[0, self.NREC-1, self.NSTEP*self.DELTAT*1000, 0])
        plt.colorbar(label='Amplitude')
        plt.title('Horizontal Component Seismogram')
        plt.xlabel('Receiver number')
        plt.ylabel('Time (ms)')
        
        # Plot vertical component seismogram
        plt.subplot(212)
        plt.imshow(self.seismogram_vz, aspect='auto', cmap='seismic',
                  extent=[0, self.NREC-1, self.NSTEP*self.DELTAT*1000, 0])
        plt.colorbar(label='Amplitude')
        plt.title('Vertical Component Seismogram')
        plt.xlabel('Receiver number')
        plt.ylabel('Time (ms)')
        
        plt.tight_layout()
        plt.savefig('seismograms.png')
        plt.close()

    def create_color_image(self, image_data_2D, it, field_number):
        """Create a color image of a given vector component"""
        # 首先确定场名称
        if field_number == 1:
            field_name = 'Vx'
        else:
            field_name = 'Vy'
            
        # Parameters for visualization
        POWER_DISPLAY = 0.30  # non linear display to enhance small amplitudes
        cutvect = 0.01       # amplitude threshold above which we draw the color point
        WHITE_BACKGROUND = True
        width_cross = 5      # size of cross and square in pixels
        thickness_cross = 1
        
        # 使用正确的field_name构建文件名
        fig_name = os.path.join(self.output_dir, f'image{it:06d}_{field_name}.png')
        
        # 计算有效区域的范围（不包括PML边界）
        x_start = self.NPOINTS_PML if self.USE_PML_XMIN else 0
        x_end = self.NX - self.NPOINTS_PML if self.USE_PML_XMAX else self.NX
        y_start = self.NPOINTS_PML if self.USE_PML_YMIN else 0
        y_end = self.NY - self.NPOINTS_PML if self.USE_PML_YMAX else self.NY
        
        # 创建新的图形和子图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 只显示有效区域的数据
        valid_data = image_data_2D[x_start:x_end, y_start:y_end]
        max_amplitude = np.max(np.abs(valid_data))
        
        # 计算显示范围（转换为厘米）
        extent = [
            x_start * self.DELTAX * 100,  # 左边界
            x_end * self.DELTAX * 100,    # 右边界
            y_start * self.DELTAY * 100,  # 下边界
            y_end * self.DELTAY * 100     # 上边界
        ]
        
        # 使用imshow显示图像（只显示有效区域）
        im = ax.imshow(valid_data.T, extent=extent, cmap='seismic', 
                      vmin=-max_amplitude, vmax=max_amplitude)
        
        # 添加colorbar
        plt.colorbar(im, ax=ax, label='Amplitude')
        
        # 添加标题和轴标签
        ax.set_title(f'Wave field snapshot - {field_name} at t = {it*self.DELTAT*1000:.2f} ms')
        ax.set_xlabel('Distance (cm)')
        ax.set_ylabel('Distance (cm)')
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 设置边框
        ax.set_box_aspect(1)  # 保持方形比例
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        
        # 保存图像到正确的输出目录
        plt.savefig(fig_name, bbox_inches='tight', dpi=300)
        plt.close()

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
        quasi_cp_max = cp.maximum(cp.sqrt(self.c33/self.rho), cp.sqrt(self.c11/self.rho))
        Courant_number = quasi_cp_max * self.DELTAT * cp.sqrt(1.0/self.DELTAX**2 + 1.0/self.DELTAY**2)
        print(f'Courant number is {float(Courant_number)}')
        if Courant_number > 1.0:
            raise ValueError('Time step is too large, simulation will be unstable')
        
        # Setup PML boundaries
        self.setup_pml_boundary()
        
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