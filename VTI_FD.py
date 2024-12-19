import numpy as np
from dataclasses import dataclass
from typing import Tuple
import matplotlib.pyplot as plt
import time
import cProfile

@dataclass
class SimulationConfig:
    """模拟配置参数类"""
    # 网格参数
    NX: int = 401
    NY: int = 401
    DELTAX: float = 0.0625e-2
    DELTAY: float = 0.0625e-2
    
    # PML参数
    USE_PML_XMIN: bool = True
    USE_PML_XMAX: bool = True 
    USE_PML_YMIN: bool = True
    USE_PML_YMAX: bool = True
    NPOINTS_PML: int = 10
    
    # 材料参数(Model I from Becache)
    c11: float = 4.0e10 
    c13: float = 3.8e10 
    c33: float = 20.0e10 
    c44: float = 2.0e10 
    rho: float = 4000.0
    
    # 时间步长参数
    NSTEP: int = 3000
    DELTAT: float = 50e-9
    
    # 震源参数
    f0: float = 200e3
    t0: float = 1.20 / f0
    factor: float = 1e7
    ISOURCE: int = NX // 2
    JSOURCE: int = NY // 2
    ANGLE_FORCE: float = 0.0
    
    # 显示参数
    IT_DISPLAY: int = 100
    
    # PML参数
    NPOWER: float = 2.0
    K_MAX_PML: float = 1.0
    ALPHA_MAX_PML: float = 2.0 * np.pi * (f0/2.0)
    
    # 添加稳定性阈值
    STABILITY_THRESHOLD: float = 1e25
    
    PI: float = np.pi
    
    AXISYM: bool = False
    
    def __post_init__(self):
        """计算源位置"""
        self.xsource = (self.ISOURCE - 1) * self.DELTAX
        self.ysource = (self.JSOURCE - 1) * self.DELTAY
        
    def validate_parameters(self) -> None:
        """验证参数的有效性"""
        # 检查正交各向异性材料的定义
        if self.c11 * self.c33 - self.c13 * self.c13 <= 0:
            raise ValueError("Problem in definition of orthotropic material")
            
        # 检查PML模型的数学稳定性
        aniso_stability_criterion = ((self.c13 + self.c44)**2 - self.c11*(self.c33-self.c44)) * \
                                  ((self.c13 + self.c44)**2 + self.c44*(self.c33-self.c44))
        
        if aniso_stability_criterion > 0 and any([self.USE_PML_XMIN, self.USE_PML_XMAX, 
                                                self.USE_PML_YMIN, self.USE_PML_YMAX]):
            print("WARNING: PML model mathematically intrinsically unstable for this anisotropic material")
            
        # 检查Courant稳定性条件
        quasi_cp_max = max(np.sqrt(self.c33/self.rho), np.sqrt(self.c11/self.rho))
        courant_number = quasi_cp_max * self.DELTAT * np.sqrt(1/self.DELTAX**2 + 1/self.DELTAY**2)
        
        if courant_number > 1.0:
            raise ValueError("Time step is too large, simulation will be unstable")
            
        return quasi_cp_max 

class PMLCalculator:
    """PML(完美匹配层)计算器"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.initialize_pml_arrays()
        
    def initialize_pml_arrays(self) -> Tuple[np.ndarray, ...]:
        """初始化PML相关的数组"""
        # 初始化x方向的数组
        self.d_x = np.zeros(self.config.NX)
        self.d_x_half = np.zeros(self.config.NX)
        self.K_x = np.ones(self.config.NX)
        self.K_x_half = np.ones(self.config.NX)
        self.alpha_x = np.zeros(self.config.NX)
        self.alpha_x_half = np.zeros(self.config.NX)
        self.a_x = np.zeros(self.config.NX)
        self.a_x_half = np.zeros(self.config.NX)
        self.b_x = np.zeros(self.config.NX)
        self.b_x_half = np.zeros(self.config.NX)

        # 初始化y方向的数组
        self.d_y = np.zeros(self.config.NY)
        self.d_y_half = np.zeros(self.config.NY)
        self.K_y = np.ones(self.config.NY)
        self.K_y_half = np.ones(self.config.NY)
        self.alpha_y = np.zeros(self.config.NY)
        self.alpha_y_half = np.zeros(self.config.NY)
        self.a_y = np.zeros(self.config.NY)
        self.a_y_half = np.zeros(self.config.NY)
        self.b_y = np.zeros(self.config.NY)
        self.b_y_half = np.zeros(self.config.NY)
        
    def compute_pml_parameters(self, quasi_cp_max: float) -> None:
        """计算PML参数"""
        # 计算PML层的厚度
        thickness_PML_x = self.config.NPOINTS_PML * self.config.DELTAX
        thickness_PML_y = self.config.NPOINTS_PML * self.config.DELTAY
        
        # 计算反射系数
        Rcoef = 0.001
        
        # 计算d0
        d0_x = -(self.config.NPOWER + 1) * quasi_cp_max * np.log(Rcoef) / (2.0 * thickness_PML_x)
        d0_y = -(self.config.NPOWER + 1) * quasi_cp_max * np.log(Rcoef) / (2.0 * thickness_PML_y)
        
        # 计算PML边界的原点
        xoriginleft = thickness_PML_x
        xoriginright = (self.config.NX-1) * self.config.DELTAX - thickness_PML_x
        yoriginbottom = thickness_PML_y
        yorigintop = (self.config.NY-1) * self.config.DELTAY - thickness_PML_y
        
        # 计算x方向的PML参数
        self._compute_x_pml_profiles(d0_x, thickness_PML_x, xoriginleft, xoriginright)
        
        # 计算y方向的PML参数
        self._compute_y_pml_profiles(d0_y, thickness_PML_y, yoriginbottom, yorigintop)
        
    def _compute_x_pml_profiles(self, d0_x: float, thickness_PML_x: float, 
                               xoriginleft: float, xoriginright: float) -> None:
        """计算x方向的PML剖面"""
        for i in range(self.config.NX):
            xval = self.config.DELTAX * i
            
            # 左边界
            if self.config.USE_PML_XMIN:
                self._compute_pml_profile_point(
                    xoriginleft - xval,
                    thickness_PML_x,
                    d0_x,
                    i,
                    'x'
                )
                self._compute_pml_profile_half(
                    xoriginleft - (xval + self.config.DELTAX/2.0),
                    thickness_PML_x,
                    d0_x,
                    i,
                    'x'
                )
                
            # 右边界
            if self.config.USE_PML_XMAX:
                self._compute_pml_profile_point(
                    xval - xoriginright,
                    thickness_PML_x,
                    d0_x,
                    i,
                    'x'
                )
                self._compute_pml_profile_half(
                    xval + self.config.DELTAX/2.0 - xoriginright,
                    thickness_PML_x,
                    d0_x,
                    i,
                    'x'
                )
                
            # 计算b和a系数
            self._compute_absorption_coefficients(i, 'x')
            
    def _compute_y_pml_profiles(self, d0_y: float, thickness_PML_y: float,
                               yoriginbottom: float, yorigintop: float) -> None:
        """计算y方向的PML剖面"""
        for j in range(self.config.NY):
            yval = self.config.DELTAY * j
            
            # 底部边界
            if self.config.USE_PML_YMIN:
                self._compute_pml_profile_point(
                    yoriginbottom - yval,
                    thickness_PML_y,
                    d0_y,
                    j,
                    'y'
                )
                self._compute_pml_profile_half(
                    yoriginbottom - (yval + self.config.DELTAY/2.0),
                    thickness_PML_y,
                    d0_y,
                    j,
                    'y'
                )
                
            # 顶部边界
            if self.config.USE_PML_YMAX:
                self._compute_pml_profile_point(
                    yval - yorigintop,
                    thickness_PML_y,
                    d0_y,
                    j,
                    'y'
                )
                self._compute_pml_profile_half(
                    yval + self.config.DELTAY/2.0 - yorigintop,
                    thickness_PML_y,
                    d0_y,
                    j,
                    'y'
                )
                
            # 计算b和a系数
            self._compute_absorption_coefficients(j, 'y')
            
    def _compute_pml_profile_point(self, abscissa_in_PML: float, thickness_PML: float,
                                  d0: float, idx: int, direction: str) -> None:
        """计算PML剖面的网格点值"""
        if abscissa_in_PML >= 0:
            abscissa_normalized = abscissa_in_PML / thickness_PML
            if direction == 'x':
                if self.config.AXISYM:
                    self.d_x[idx] = d0 * (abscissa_normalized**self.config.NPOWER) * \
                                   (1.0 + 1.5 * np.log(1.0 / (1.0 - abscissa_normalized)))
                else:
                    self.d_x[idx] = d0 * abscissa_normalized**self.config.NPOWER
                self.K_x[idx] = 1.0 + (self.config.K_MAX_PML - 1.0) * abscissa_normalized**self.config.NPOWER
                self.alpha_x[idx] = self.config.ALPHA_MAX_PML * (1.0 - abscissa_normalized)
            else:
                self.d_y[idx] = d0 * abscissa_normalized**self.config.NPOWER
                self.K_y[idx] = 1.0 + (self.config.K_MAX_PML - 1.0) * abscissa_normalized**self.config.NPOWER
                self.alpha_y[idx] = self.config.ALPHA_MAX_PML * (1.0 - abscissa_normalized)
                
    def _compute_pml_profile_half(self, abscissa_in_PML: float, thickness_PML: float,
                                 d0: float, idx: int, direction: str) -> None:
        """计算PML剖面的半网格点值"""
        if abscissa_in_PML >= 0:
            abscissa_normalized = abscissa_in_PML / thickness_PML
            if direction == 'x':
                self.d_x_half[idx] = d0 * abscissa_normalized**self.config.NPOWER
                self.K_x_half[idx] = 1.0 + (self.config.K_MAX_PML - 1.0) * abscissa_normalized**self.config.NPOWER
                self.alpha_x_half[idx] = self.config.ALPHA_MAX_PML * (1.0 - abscissa_normalized)
            else:
                self.d_y_half[idx] = d0 * abscissa_normalized**self.config.NPOWER
                self.K_y_half[idx] = 1.0 + (self.config.K_MAX_PML - 1.0) * abscissa_normalized**self.config.NPOWER
                self.alpha_y_half[idx] = self.config.ALPHA_MAX_PML * (1.0 - abscissa_normalized)
                
    def _compute_absorption_coefficients(self, idx: int, direction: str) -> None:
        """计算吸收系数"""
        if direction == 'x':
            # 确保alpha不为负
            self.alpha_x[idx] = max(0, self.alpha_x[idx])
            self.alpha_x_half[idx] = max(0, self.alpha_x_half[idx])
            
            # 计算b系数
            self.b_x[idx] = np.exp(-(self.d_x[idx] / self.K_x[idx] + self.alpha_x[idx]) * self.config.DELTAT)
            self.b_x_half[idx] = np.exp(-(self.d_x_half[idx] / self.K_x_half[idx] + self.alpha_x_half[idx]) * self.config.DELTAT)
            
            # 计算a系数
            if abs(self.d_x[idx]) > 1e-6:
                self.a_x[idx] = self.d_x[idx] * (self.b_x[idx] - 1.0) / \
                               (self.K_x[idx] * (self.d_x[idx] + self.K_x[idx] * self.alpha_x[idx]))
            if abs(self.d_x_half[idx]) > 1e-6:
                self.a_x_half[idx] = self.d_x_half[idx] * (self.b_x_half[idx] - 1.0) / \
                                    (self.K_x_half[idx] * (self.d_x_half[idx] + self.K_x_half[idx] * self.alpha_x_half[idx]))
        else:
            # y方向的计算与x方向类似
            self.alpha_y[idx] = max(0, self.alpha_y[idx])
            self.alpha_y_half[idx] = max(0, self.alpha_y_half[idx])
            
            self.b_y[idx] = np.exp(-(self.d_y[idx] / self.K_y[idx] + self.alpha_y[idx]) * self.config.DELTAT)
            self.b_y_half[idx] = np.exp(-(self.d_y_half[idx] / self.K_y_half[idx] + self.alpha_y_half[idx]) * self.config.DELTAT)
            
            if abs(self.d_y[idx]) > 1e-6:
                self.a_y[idx] = self.d_y[idx] * (self.b_y[idx] - 1.0) / \
                               (self.K_y[idx] * (self.d_y[idx] + self.K_y[idx] * self.alpha_y[idx]))
            if abs(self.d_y_half[idx]) > 1e-6:
                self.a_y_half[idx] = self.d_y_half[idx] * (self.b_y_half[idx] - 1.0) / \
                                    (self.K_y_half[idx] * (self.d_y_half[idx] + self.K_y_half[idx] * self.alpha_y_half[idx]))

class WaveFieldSimulator:
    """波场模拟器"""
    
    def __init__(self, config: SimulationConfig, pml: PMLCalculator):
        self.config = config
        self.pml = pml
        self.initialize_arrays()
        
    def initialize_arrays(self):
        """初始化波场数组"""
        # 主要波场数组
        self.vx = np.zeros((self.config.NX, self.config.NY))
        self.vy = np.zeros((self.config.NX, self.config.NY))
        self.sigmaxx = np.zeros((self.config.NX, self.config.NY))
        self.sigmayy = np.zeros((self.config.NX, self.config.NY))
        self.sigmaxy = np.zeros((self.config.NX, self.config.NY))
        
        # PML内存变量
        self.memory_dvx_dx = np.zeros((self.config.NX, self.config.NY))
        self.memory_dvx_dy = np.zeros((self.config.NX, self.config.NY))
        self.memory_dvy_dx = np.zeros((self.config.NX, self.config.NY))
        self.memory_dvy_dy = np.zeros((self.config.NX, self.config.NY))
        self.memory_dsigmaxx_dx = np.zeros((self.config.NX, self.config.NY))
        self.memory_dsigmayy_dy = np.zeros((self.config.NX, self.config.NY))
        self.memory_dsigmaxy_dx = np.zeros((self.config.NX, self.config.NY))
        self.memory_dsigmaxy_dy = np.zeros((self.config.NX, self.config.NY))

    def compute_stress(self):
        """计算应力分量"""
        # 计算sigmaxx和sigmayy
        for j in range(1, self.config.NY):
            for i in range(self.config.NX-1):
                value_dvx_dx = (self.vx[i+1,j] - self.vx[i,j]) / self.config.DELTAX
                value_dvy_dy = (self.vy[i,j] - self.vy[i,j-1]) / self.config.DELTAY
                
                self.memory_dvx_dx[i,j] = self.pml.b_x_half[i] * self.memory_dvx_dx[i,j] + \
                                     self.pml.a_x_half[i] * value_dvx_dx
                self.memory_dvy_dy[i,j] = self.pml.b_y[j] * self.memory_dvy_dy[i,j] + \
                                     self.pml.a_y[j] * value_dvy_dy
                
                value_dvx_dx = value_dvx_dx / self.pml.K_x_half[i] + self.memory_dvx_dx[i,j]
                value_dvy_dy = value_dvy_dy / self.pml.K_y[j] + self.memory_dvy_dy[i,j]
                
                self.sigmaxx[i,j] += (self.config.c11 * value_dvx_dx + 
                                 self.config.c13 * value_dvy_dy) * self.config.DELTAT
                self.sigmayy[i,j] += (self.config.c13 * value_dvx_dx + 
                                 self.config.c33 * value_dvy_dy) * self.config.DELTAT
        
        # 计算sigmaxy
        for j in range(self.config.NY-1):
            for i in range(1, self.config.NX):
                value_dvy_dx = (self.vy[i,j] - self.vy[i-1,j]) / self.config.DELTAX
                value_dvx_dy = (self.vx[i,j+1] - self.vx[i,j]) / self.config.DELTAY
                
                self.memory_dvy_dx[i,j] = self.pml.b_x[i] * self.memory_dvy_dx[i,j] + \
                                     self.pml.a_x[i] * value_dvy_dx
                self.memory_dvx_dy[i,j] = self.pml.b_y_half[j] * self.memory_dvx_dy[i,j] + \
                                     self.pml.a_y_half[j] * value_dvx_dy
                
                value_dvy_dx = value_dvy_dx / self.pml.K_x[i] + self.memory_dvy_dx[i,j]
                value_dvx_dy = value_dvx_dy / self.pml.K_y_half[j] + self.memory_dvx_dy[i,j]
                
                self.sigmaxy[i,j] += self.config.c44 * (value_dvy_dx + value_dvx_dy) * self.config.DELTAT
    
    def compute_velocity(self):
        """计算速度分量"""
        # 计算vx
        for j in range(1, self.config.NY):
            for i in range(1, self.config.NX):
                value_dsigmaxx_dx = (self.sigmaxx[i,j] - self.sigmaxx[i-1,j]) / self.config.DELTAX
                value_dsigmaxy_dy = (self.sigmaxy[i,j] - self.sigmaxy[i,j-1]) / self.config.DELTAY
                
                self.memory_dsigmaxx_dx[i,j] = self.pml.b_x[i] * self.memory_dsigmaxx_dx[i,j] + \
                                              self.pml.a_x[i] * value_dsigmaxx_dx
                self.memory_dsigmaxy_dy[i,j] = self.pml.b_y[j] * self.memory_dsigmaxy_dy[i,j] + \
                                              self.pml.a_y[j] * value_dsigmaxy_dy
                
                value_dsigmaxx_dx = value_dsigmaxx_dx / self.pml.K_x[i] + self.memory_dsigmaxx_dx[i,j]
                value_dsigmaxy_dy = value_dsigmaxy_dy / self.pml.K_y[j] + self.memory_dsigmaxy_dy[i,j]
                
                self.vx[i,j] += (value_dsigmaxx_dx + value_dsigmaxy_dy) * self.config.DELTAT / self.config.rho
        
        # 计算vy
        for j in range(self.config.NY-1):
            for i in range(self.config.NX-1):
                value_dsigmaxy_dx = (self.sigmaxy[i+1,j] - self.sigmaxy[i,j]) / self.config.DELTAX
                value_dsigmayy_dy = (self.sigmayy[i,j+1] - self.sigmayy[i,j]) / self.config.DELTAY
                
                self.memory_dsigmaxy_dx[i,j] = self.pml.b_x_half[i] * self.memory_dsigmaxy_dx[i,j] + \
                                              self.pml.a_x_half[i] * value_dsigmaxy_dx
                self.memory_dsigmayy_dy[i,j] = self.pml.b_y_half[j] * self.memory_dsigmayy_dy[i,j] + \
                                              self.pml.a_y_half[j] * value_dsigmayy_dy
                
                value_dsigmaxy_dx = value_dsigmaxy_dx / self.pml.K_x_half[i] + self.memory_dsigmaxy_dx[i,j]
                value_dsigmayy_dy = value_dsigmayy_dy / self.pml.K_y_half[j] + self.memory_dsigmayy_dy[i,j]
                
                self.vy[i,j] += (value_dsigmaxy_dx + value_dsigmayy_dy) * self.config.DELTAT / self.config.rho
                
    def add_source(self, it: int):
        """添加震源"""
        a = np.pi * np.pi * self.config.f0 * self.config.f0
        t = (it - 1) * self.config.DELTAT
        
        # 使用高斯一阶导数作为震源函数
        source_term = -self.config.factor * 2.0 * a * (t - self.config.t0) * \
                     np.exp(-a * (t - self.config.t0)**2)
        
        force_x = np.sin(self.config.ANGLE_FORCE * np.pi / 180.0) * source_term
        force_y = np.cos(self.config.ANGLE_FORCE * np.pi / 180.0) * source_term
        
        self.vx[self.config.ISOURCE, self.config.JSOURCE] += force_x * self.config.DELTAT / self.config.rho
        self.vy[self.config.ISOURCE, self.config.JSOURCE] += force_y * self.config.DELTAT / self.config.rho
        
    def apply_boundary_conditions(self):
        """应用边界条件（刚性边界）"""
        # x方向边界
        self.vx[0,:] = 0.0
        self.vx[-1,:] = 0.0
        self.vx[:,0] = 0.0
        self.vx[:,-1] = 0.0
        
        # y方向边界
        self.vy[0,:] = 0.0
        self.vy[-1,:] = 0.0
        self.vy[:,0] = 0.0
        self.vy[:,-1] = 0.0
        
    def check_stability(self, it: int) -> float:
        """检查数值稳定性"""
        velocnorm = np.max(np.sqrt(self.vx**2 + self.vy**2))
        
        if velocnorm > 1e25:  # STABILITY_THRESHOLD
            raise RuntimeError('代码变得不稳定并发散')
            
        if it % self.config.IT_DISPLAY == 0 or it == 5:
            print(f'Time step # {it} out of {self.config.NSTEP}')
            print(f'Time: {(it-1)*self.config.DELTAT:.6f} seconds')
            print(f'Max norm velocity vector V (m/s) = {velocnorm}')
            print()
            
        return velocnorm

class WaveFieldVisualizer:
    """波场可视化器"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        # 非线性显示以增强小振幅
        self.POWER_DISPLAY = 0.30
        # 绘制颜色点的振幅阈值
        self.cutvect = 0.01
        # 使用白色背景
        self.WHITE_BACKGROUND = True
        # 源和接收器的十字和方块大小（像素）
        self.width_cross = 5
        self.thickness_cross = 1
        self.size_square = 3
        
    def create_color_image(self, image_data_2D: np.ndarray, it: int, field_number: int):
        """创建波场的彩色图像
        
        完全按照原始Fortran代码的逻辑实现
        """
        # 创建文件名
        if field_number == 1:
            filename = f'image{it:06d}_Vx.png'
        else:
            filename = f'image{it:06d}_Vy.png'
            
        # 计算最大振幅
        max_amplitude = np.max(np.abs(image_data_2D))
        
        # 创建图像数组 (NY x NX x 3)
        img = np.zeros((self.config.NY, self.config.NX, 3), dtype=np.uint8)
        
        # 图像从左上角开始
        for iy in range(self.config.NY-1, -1, -1):
            for ix in range(self.config.NX):
                # 归一化数据到[-1,1]
                normalized_value = image_data_2D[ix,iy] / max_amplitude
                
                # 抑制[-1,+1]之外的值以避免小的边缘效应
                normalized_value = np.clip(normalized_value, -1.0, 1.0)
                
                # 绘制表示源的橙色十字
                if ((ix >= self.config.ISOURCE - self.width_cross and 
                     ix <= self.config.ISOURCE + self.width_cross and
                     iy >= self.config.JSOURCE - self.thickness_cross and 
                     iy <= self.config.JSOURCE + self.thickness_cross) or
                    (ix >= self.config.ISOURCE - self.thickness_cross and 
                     ix <= self.config.ISOURCE + self.thickness_cross and
                     iy >= self.config.JSOURCE - self.width_cross and 
                     iy <= self.config.JSOURCE + self.width_cross)):
                    R, G, B = 255, 157, 0
                    
                # 在图像周围显示两个像素厚的黑色边框
                elif (ix <= 2 or ix >= self.config.NX-1 or 
                      iy <= 2 or iy >= self.config.NY-1):
                    R, G, B = 0, 0, 0
                    
                # 显示PML层的边缘
                elif ((self.config.USE_PML_XMIN and ix == self.config.NPOINTS_PML) or
                      (self.config.USE_PML_XMAX and ix == self.config.NX - self.config.NPOINTS_PML) or
                      (self.config.USE_PML_YMIN and iy == self.config.NPOINTS_PML) or
                      (self.config.USE_PML_YMAX and iy == self.config.NY - self.config.NPOINTS_PML)):
                    R, G, B = 255, 150, 0
                    
                # 抑制所有低于阈值的值
                elif abs(image_data_2D[ix,iy]) <= max_amplitude * self.cutvect:
                    # 对低于阈值的点使用黑色或白色背景
                    if self.WHITE_BACKGROUND:
                        R, G, B = 255, 255, 255
                    else:
                        R, G, B = 0, 0, 0
                        
                # 使用红色表示正值，蓝色表示负值
                elif normalized_value >= 0.0:
                    R = int(255.0 * normalized_value**self.POWER_DISPLAY)
                    G, B = 0, 0
                else:
                    R, G = 0, 0
                    B = int(255.0 * abs(normalized_value)**self.POWER_DISPLAY)
                    
                # 设置像素颜色
                img[self.config.NY-1-iy, ix] = [R, G, B]
                
        # 保存图像
        plt.imsave(filename, img)

def run_simulation():
    """运行模拟"""
    start_time = time.time()

    # 使用cProfile进行性能分析
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        # 创建配置
        config = SimulationConfig()
        
        # 验证参数并获取最大P波速度
        quasi_cp_max = config.validate_parameters()
        
        # 初始化PML
        pml = PMLCalculator(config)
        pml.compute_pml_parameters(quasi_cp_max)
        
        # 初始化波场模拟器
        simulator = WaveFieldSimulator(config, pml)
        
        # 初始化可视化器
        visualizer = WaveFieldVisualizer(config)
        
        # 时间迭代
        for it in range(1, config.NSTEP + 1):
            # 计算应力
            simulator.compute_stress()
            
            # 计算速度
            simulator.compute_velocity()
            
            # 添加震源
            simulator.add_source(it)
            
            # 应用边界条件
            simulator.apply_boundary_conditions()
            
            # 检查稳定性并可能输出信息
            velocnorm = simulator.check_stability(it)
            
            # 可视化（如果是输出时间步）
            if it % config.IT_DISPLAY == 0 or it == 5:
                visualizer.create_color_image(simulator.vx, it, 1)
                visualizer.create_color_image(simulator.vy, it, 2)

    finally:
        # 输出性能分析结果
        profiler.disable()
        print("\n性能分析结果:")
        profiler.print_stats(sort='cumulative')
        print(f"\n总运行时间: {time.time() - start_time:.2f} 秒")     
   
    print("\n模拟结束\n")

if __name__ == "__main__":
    run_simulation()