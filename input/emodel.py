import numpy as np

# 定义nz和nx的值
nz=201
nx=801

# 创建一个5*500*500的数组，初始值为0
model0 = np.zeros((5, nz, nx), dtype=np.float32)

# 填充数组
model0[0, :100, :] = 2500  # vp参数在1:100范围内设为2
model0[0, 100:, :] = 4000  # vp参数在200:end范围内设为5
model0[1, :100, :] = 1500 # vs参数在1:100范围内设为1
model0[1, 100:, :] = 2000   # vs参数在200:end范围内设为3
model0[2, :100, :] = 1000  # rho参数在1:100范围内设为2000
model0[2, 100:, :] = 2000   # rho参数在200:end范围内设为3000
model0[3, :100, :] = 0.3  # delta参数在1:100范围内设为0.1
model0[3, 100:, :] = 0.1   # delta参数在200:end范围内设为0.3
model0[4, :100, :] = 0.2  # epsilon参数在1:100范围内设为0.05
model0[4, 100:, :] = 0.15   # epsilon参数在200:end范围内设为0.15    

model = np.copy(model0)
model[0,70:80,375:426] = 3500
model[1,70:80,375:426] = 1750
model[2,70:80,375:426] = 1750
model[3,70:80,375:426] = 0.35
model[4,70:80,375:426] = 0.175

# 保存数组为xyz文件
np.savetxt('model0.xyz', model0.reshape(-1, model0.shape[-1]), fmt='%.6f')
np.savetxt('model.xyz', model.reshape(-1, model.shape[-1]), fmt='%.6f')



