import numpy as np
import matplotlib.pyplot as plt

try:
    # 读取地震记录数据
    vx_data = np.loadtxt('seismogram_vx.dat')
    vy_data = np.loadtxt('seismogram_vy.dat')
except FileNotFoundError:
    print("错误：找不到数据文件！请确保 seismogram_vx.dat 和 seismogram_vy.dat 文件存在。")
    exit()
except ValueError:
    print("错误：数据文件格式不正确！请确保文件包含有效的数值数据。")
    exit()

# 设置时间和距离参数
nt, nr = vx_data.shape
dt = 50e-9  # 时间步长(与Fortran代码中的DELTAT一致)
dx = 4      # 检波器间距(与Fortran代码中的rec_dx一致)
t = np.arange(nt) * dt
x = np.arange(nr) * dx

# 创建图形
plt.figure(figsize=(12, 8))

# 绘制水平分量地震记录
plt.subplot(211)
plt.imshow(vx_data, aspect='auto', cmap='gray', 
           extent=[x[0], x[-1], t[-1], t[0]])
plt.colorbar(label='Amplitude')
plt.title('Horizontal Component (Vx)')
plt.xlabel('Distance (m)')
plt.ylabel('Time (s)')

# 绘制垂直分量地震记录
plt.subplot(212)
plt.imshow(vy_data, aspect='auto', cmap='gray',
           extent=[x[0], x[-1], t[-1], t[0]])
plt.colorbar(label='Amplitude')
plt.title('Vertical Component (Vy)')
plt.xlabel('Distance (m)')
plt.ylabel('Time (s)')

plt.tight_layout()
plt.savefig('seismograms.png', dpi=300, bbox_inches='tight')
plt.close()