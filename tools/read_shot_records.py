import numpy as np
import matplotlib.pyplot as plt

# 读取数据
vx_records = np.load('shot_records_vx.npy') 
vz_records = np.load('shot_records_vz.npy')
params = np.load('simulation_params.npy', allow_pickle=True).item()

# 打印数据信息
print("数据形状:")
print(f"水平分量记录形状: {vx_records.shape}")
print(f"垂直分量记录形状: {vz_records.shape}")
print("\n模拟参数:")
for key, value in params.items():
    print(f"{key}: {value}")

# 绘图部分修正
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(vx_records[0], aspect='auto', cmap='seismic')
plt.title('水平分量')
plt.xlabel('道号')
plt.ylabel('采样点')
plt.colorbar(label='振幅')

plt.subplot(122)
plt.imshow(vz_records[0], aspect='auto', cmap='seismic')
plt.title('垂直分量')
plt.xlabel('道号')
plt.ylabel('采样点')
plt.colorbar(label='振幅')

plt.tight_layout()
plt.show()

