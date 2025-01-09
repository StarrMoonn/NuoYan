import numpy as np

class VTIParamsCalculator:
    """VTI介质参数计算器：参数转换和稳定性检查"""
    
    def __init__(self, vp, vs, rho, epsilon, delta):
        """
        初始化参数
        
        参数:
        vp: float, P波速度 (m/s)
        vs: float, S波速度 (m/s)
        rho: float, 密度 (kg/m³)
        epsilon: float, Thomsen参数ε
        delta: float, Thomsen参数δ
        """
        self.vp = float(vp)
        self.vs = float(vs)
        self.rho = float(rho)
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        
        # 计算刚度系数
        self._compute_cij()
        
    def _compute_cij(self):
        """计算VTI介质的刚度系数 (Pa)"""
        self.c33 = self.rho * self.vp**2
        self.c44 = self.rho * self.vs**2
        self.c11 = self.c33 * (1 + 2*self.epsilon)
        temp = 2 * self.delta * self.c33 * (self.c33 - self.c44) + (self.c33 - self.c44)**2
        self.c13 = np.sqrt(temp) - self.c44
        
    def get_cij(self):
        """返回计算得到的刚度系数"""
        return {
            'c11': self.c11,
            'c13': self.c13,
            'c33': self.c33,
            'c44': self.c44,
            'rho': self.rho
        }
        
    def check_stability(self, dx, dz, dt, f0):
        """
        检查数值稳定性条件
        
        参数:
        dx, dz: float, 空间步长 (m)
        dt: float, 时间步长 (s)
        f0: float, 雷克子波主频 (Hz)
        
        返回:
        dict: 包含稳定性检查结果
        """
        # 计算最大速度
        v_max = self.vp * np.sqrt(1 + 2 * self.epsilon)
        
        # 计算Courant数
        courant = v_max * dt * np.sqrt(1/dx**2 + 1/dz**2)
        
        # 计算网格点数/波长
        min_wavelength = v_max / f0
        dmin = min(dx, dz)
        grid_points = min_wavelength / dmin
        
        # 判断稳定性
        is_stable_cfl = courant <= 1.0
        is_stable_dispersion = grid_points >= 5
        
        return {
            'v_max': v_max,
            'courant': courant,
            'grid_points': grid_points,
            'is_stable_cfl': is_stable_cfl,
            'is_stable_dispersion': is_stable_dispersion
        }
    
    def print_results(self):
        """打印计算结果"""
        print("\n=== VTI介质参数计算结果 ===")
        print(f"输入参数:")
        print(f"Vp = {self.vp:.1f} m/s")
        print(f"Vs = {self.vs:.1f} m/s")
        print(f"密度 = {self.rho:.1f} kg/m³")
        print(f"ε = {self.epsilon:.3f}")
        print(f"δ = {self.delta:.3f}")
        
        print(f"\n刚度系数 (Pa):")
        print(f"C11 = {self.c11:.3e}")
        print(f"C13 = {self.c13:.3e}")
        print(f"C33 = {self.c33:.3e}")
        print(f"C44 = {self.c44:.3e}")

def main():
    """使用示例"""
    # 1. 设置VTI介质参数
    vp = 3000.0      # m/s
    vs = 1500.0      # m/s
    rho = 2000.0     # kg/m³
    epsilon = 0.1
    delta = 0.4
    
    # 2. 创建计算器实例
    calculator = VTIParamsCalculator(vp, vs, rho, epsilon, delta)
    
    # 3. 打印参数转换结果
    calculator.print_results()
    
    # 4. 检查数值稳定性
    dx = 5.0         # m
    dz = 5.0         # m
    dt = 0.001       # s
    f0 = 20.0        # Hz
    
    stability = calculator.check_stability(dx, dz, dt, f0)
    
    print("\n=== 数值稳定性检查 ===")
    print(f"最大速度: {stability['v_max']:.1f} m/s")
    print(f"Courant数: {stability['courant']:.3f}")
    print(f"每波长网格点数: {stability['grid_points']:.1f}")
    print(f"CFL条件: {'满足' if stability['is_stable_cfl'] else '不满足'}")
    print(f"频散条件: {'满足' if stability['is_stable_dispersion'] else '不满足'}")

if __name__ == "__main__":
    main() 