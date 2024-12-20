import numpy as np
import os

def vp_vs_to_cij(vp, vs, rho, epsilon, delta):
    """
    将VTI介质的速度参数转换为刚度系数
    
    参数:
    vp: float or ndarray, P波速度 (m/s)
    vs: float or ndarray, S波速度 (m/s)
    rho: float or ndarray, 密度 (kg/m³)
    epsilon: float or ndarray, Thomsen参数 epsilon (无量纲)
    delta: float or ndarray, Thomsen参数 delta (无量纲)
    
    返回:
    tuple: (c11, c13, c33, c44)，单位为Pa
    """
    # 计算基本刚度系数
    c33 = rho * vp**2  # 垂直P波刚度系数
    c44 = rho * vs**2  # 垂直S波刚度系数
    
    # 使用Thomsen参数计算其他刚度系数
    c11 = c33 * (1 + 2*epsilon)  # 水平P波刚度系数
    
    # 计算c13
    # 使用delta参数的定义公式推导
    # delta = ((c13 + c44)**2 - (c33 - c44)**2) / (2*c33*(c33 - c44))
    # 解这个方程得到c13
    term1 = 2 * delta * c33 * (c33 - c44)
    term2 = (c33 - c44)**2
    c13 = np.sqrt(term1 + term2) - c44
    
    return c11, c13, c33, c44

def print_results(vp, vs, rho, epsilon, delta):
    """打印输入参数和计算结果"""
    print("\n输入参数:")
    print(f"VP = {vp:.2f} m/s")
    print(f"VS = {vs:.2f} m/s")
    print(f"密度 = {rho:.2f} kg/m³")
    print(f"epsilon = {epsilon:.4f}")
    print(f"delta = {delta:.4f}")
    
    c11, c13, c33, c44 = vp_vs_to_cij(vp, vs, rho, epsilon, delta)
    
    print("\n计算得到的刚度系数:")
    print(f"c11 = {c11/1e9:.2f} GPa ({c11:.2e} Pa)")
    print(f"c13 = {c13/1e9:.2f} GPa ({c13:.2e} Pa)")
    print(f"c33 = {c33/1e9:.2f} GPa ({c33:.2e} Pa)")
    print(f"c44 = {c44/1e9:.2f} GPa ({c44:.2e} Pa)")

def model_to_cij(model):
    """
    将模型参数转换为刚度系数数组
    
    参数:
    model: shape为(5, NY, NX)的数组
    model[0] = vp
    model[1] = vs
    model[2] = rho
    model[3] = delta
    model[4] = epsilon
    
    返回:
    cij_model: shape为(5, NY, NX)的数组
    cij_model[0] = c11 (Pa)
    cij_model[1] = c13 (Pa)
    cij_model[2] = c33 (Pa)
    cij_model[3] = c44 (Pa)
    cij_model[4] = rho (kg/m³)
    """
    vp = model[0]
    vs = model[1]
    rho = model[2]
    delta = model[3]
    epsilon = model[4]
    
    # 创建输出数组
    NY, NX = model.shape[1:]
    cij_model = np.zeros((5, NY, NX))
    
    # 计算刚度系数
    c33 = rho * vp**2
    c44 = rho * vs**2
    c11 = c33 * (1 + 2*epsilon)
    term1 = 2 * delta * c33 * (c33 - c44)
    term2 = (c33 - c44)**2
    c13 = np.sqrt(term1 + term2) - c44
    
    # 填充输出数组
    cij_model[0] = c11
    cij_model[1] = c13
    cij_model[2] = c33
    cij_model[3] = c44
    cij_model[4] = rho
    
    return cij_model

if __name__ == "__main__":
    # 定义网格大小
    NY, NX = 201, 801
    
    # 创建并初始化model数组
    model = np.zeros((5, NY, NX))
    
    # 设置模型参数
    # model[0] = vp
    model[0,:100,:] = 2500 
    model[0,100:,:] = 4000
    
    # model[1] = vs
    model[1,:100,:] = 1500 
    model[1,100:,:] = 2000
    
    # model[2] = rho
    model[2,:100,:] = 1000 
    model[2,100:,:] = 2000
    
    # model[3] = delta
    model[3,:100,:] = 0.15 
    model[3,100:,:] = 0.075 
    
    # model[4] = epsilon
    model[4,:100,:] = 0.25 
    model[4,100:,:] = 0.1 

    # 计算刚度系数
    cij_model = model_to_cij(model)
    
    # 创建输出目录（如果不存在）
    os.makedirs('output', exist_ok=True)
    
    # 保存数组到npy文件
    output_path = os.path.join('output', 'model.npy')
    np.save(output_path, cij_model)
    print(f"\n模型数据已保存至: {output_path}")