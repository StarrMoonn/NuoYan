import numpy as np
import matplotlib.pyplot as plt

def plot_seismogram(source_id=0, output_dir='.', save_fig=True):
    """
    读取并绘制指定震源的地震记录
    
    Args:
        source_id: 震源序号
        output_dir: 数据文件所在目录
        save_fig: 是否保存图像
    """
    # 读取数据
    vx_records = np.load(f'{output_dir}/seismogram_vx.npy')
    vy_records = np.load(f'{output_dir}/seismogram_vy.npy')
    metadata = np.load(f'{output_dir}/seismogram_metadata.npy', allow_pickle=True).item()
    
    # 获取元数据
    dt = metadata['dt']
    nt = metadata['nt']
    receiver_positions = metadata['receiver_positions']
    
    # 创建时间和距离轴
    time = np.arange(nt) * dt
    distances = np.array([pos[0] for pos in receiver_positions])  # 假设x方向是偏移距
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制水平分量
    vx_plot = ax1.imshow(vx_records[source_id], 
                        aspect='auto',
                        extent=[0, time[-1], distances[-1], distances[0]],
                        cmap='seismic',
                        interpolation='bilinear')
    plt.colorbar(vx_plot, ax=ax1)
    ax1.set_title(f'水平分量地震记录 (震源 #{source_id})')
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('偏移距 (m)')
    
    # 绘制垂直分量
    vy_plot = ax2.imshow(vy_records[source_id],
                        aspect='auto',
                        extent=[0, time[-1], distances[-1], distances[0]],
                        cmap='seismic',
                        interpolation='bilinear')
    plt.colorbar(vy_plot, ax=ax2)
    ax2.set_title(f'垂直分量地震记录 (震源 #{source_id})')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('偏移距 (m)')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    if save_fig:
        plt.savefig(f'{output_dir}/seismogram_source_{source_id}.png', dpi=300, bbox_inches='tight')
        
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 绘制第0号震源的地震记录
    plot_seismogram(source_id=0)
    """  # 绘制第0号震源的地震记录
    plot_seismogram(source_id=0)
    # 指定数据目录并绘制第1号震源的地震记录
    plot_seismogram(source_id=1, output_dir='results')
    # 绘制但不保存图像
    plot_seismogram(source_id=0, save_fig=False) """