import numpy as np

# 设置网格大小
nx = 401
nz = 401

# 创建参数数组
c11 = np.full((nz, nx), 4.0e10)    # c11 = 4.0e10 Pa
c12 = np.full((nz, nx), 3.8e10)    # c12 = 3.8e10 Pa
c22 = np.full((nz, nx), 20.0e10)   # c22 = 20.0e10 Pa
c33 = np.full((nz, nx), 2.0e10)    # c33 = 2.0e10 Pa
rho = np.full((nz, nx), 4000.0)    # rho = 4000 kg/m³

# 保存为npz文件
np.savez('material_params.npz',
         c11=c11,
         c12=c12,
         c22=c22,
         c33=c33,
         rho=rho)

# 验证保存的数据
data = np.load('material_params.npz')
print("Saved parameters shape:")
print(f"c11 shape: {data['c11'].shape}, value: {data['c11'][0,0]}")
print(f"c12 shape: {data['c12'].shape}, value: {data['c12'][0,0]}")
print(f"c22 shape: {data['c22'].shape}, value: {data['c22'][0,0]}")
print(f"c33 shape: {data['c33'].shape}, value: {data['c33'][0,0]}")
print(f"rho shape: {data['rho'].shape}, value: {data['rho'][0,0]}")