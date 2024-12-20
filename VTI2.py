import numpy as np
from numba import jit, prange
import os
import matplotlib.pyplot as plt

def ricker_wavelet(f0, dt, nt, t0=0.0):
    """
    生成雷克子波
    
    Parameters:
        f0: 主频(Hz)
        dt: 采样时间间隔(s)
        nt: 采样点数
        t0: 时间偏移(s)，默认为0
    
    Returns:
        wavelet: 雷克子波时间序列
    """
    t = np.linspace(-nt//2*dt, nt//2*dt, nt) - t0
    pi2 = (np.pi*f0*t)**2
    return (1-2*pi2)*np.exp(-pi2)

def compute_cij_from_Thomsen(vp, vs, rho, epsilon, delta):
    """
    计算VTI介质的弹性常数
    
    参数:
        vp: P波速度 (m/s)
        vs: S波速度 (m/s)
        rho: 密度 (kg/m³)
        epsilon: epsilon参数
        delta: delta参数
        
    返回:
        c11, c13, c33, c44: 弹性常数 (Pa)
    """
    # 将输入转换为numpy数组以确保一致性
    vp = np.asarray(vp)
    vs = np.asarray(vs)
    rho = np.asarray(rho)
    
    # 参数验证
    if np.any(vp <= 0):
        raise ValueError("vp must be positive")
    if np.any(vs <= 0):
        raise ValueError("vs must be positive")
    if np.any(rho <= 0):
        raise ValueError("rho must be positive")
    
    c33 = rho*(vp**2)
    c44 = rho*(vs**2)
    c11 = c33*(1+2*epsilon)
    term = (c33 - c44)
    c13 = np.sqrt(term**2 + 2*delta*c33*term) - c44
    
    return c11, c13, c33, c44

#生成震源或检波器的位置坐标数组
def generate_positions(start_x, start_z, dx_s, dz_s, n_points):
    xs = start_x + np.arange(n_points)*dx_s
    zs = start_z + np.arange(n_points)*dz_s
    return xs, zs

#初始化完美匹配层(PML)的衰减系数矩阵
def initialize_pml(nx, nz, pml_size, coeff):
    damping = np.ones((nx,nz))
    for i in range(pml_size):
        factor=1-coeff*((pml_size-i)/pml_size)**2
        damping[i,:]*=factor
        damping[-i-1,:]*=factor
    for j in range(pml_size):
        factor=1-coeff*((pml_size-j)/pml_size)**2
        damping[:,j]*=factor
        damping[:,-j-1]*=factor
    return damping

#将PML衰减系数应用到波场上
@jit(nopython=True, parallel=True)
def apply_pml(u, damping, nx, nz):
    for i in prange(nx):
        for j in prange(nz):
            u[i,j]*=damping[i,j]

@jit(nopython=True, parallel=True)
def fd_update_staggered(vx, vz, sigxx, sigzz, sigxz,
                        vx_prev, vz_prev, sigxx_prev, sigzz_prev, sigxz_prev,
                        c11, c13, c33, c44, rho,
                        dt, dx, dz, nx, nz):
    """
    四阶精度交错网格有限差分更新
    """
    # 四阶中心差分系数
    c1 = 9.0/8.0   # 一阶导数的四阶精度系数
    c2 = -1.0/24.0 # 一阶导数的四阶精度系数

    # 维度检查
    if (vx.shape[0] != nx or vx.shape[1] != nz):
        raise ValueError("Array dimensions do not match nx and nz")
    
    # 1. 更新速度场 (四阶中心差分)
    for i in prange(2, nx-2):
        for j in range(2, nz-2):
            # x方向四阶差分
            dσxx_dx = (c1*(sigxx[i,j] - sigxx[i-1,j]) + 
                      c2*(sigxx[i+1,j] - sigxx[i-2,j])) / dx
            
            # z方向四阶差分
            dσxz_dz = (c1*(sigxz[i,j] - sigxz[i,j-1]) + 
                      c2*(sigxz[i,j+1] - sigxz[i,j-2])) / dz
            
            dσxz_dx = (c1*(sigxz[i+1,j] - sigxz[i,j]) + 
                      c2*(sigxz[i+2,j] - sigxz[i-1,j])) / dx
            
            dσzz_dz = (c1*(sigzz[i,j+1] - sigzz[i,j]) + 
                      c2*(sigzz[i,j+2] - sigzz[i,j-1])) / dz
            
            # 更新速度
            vx[i,j] = vx_prev[i,j] + dt * (dσxx_dx + dσxz_dz) / rho[i,j]
            vz[i,j] = vz_prev[i,j] + dt * (dσxz_dx + dσzz_dz) / rho[i,j]
    
    # 2. 更新应力场
    for i in prange(2, nx-2):
        for j in range(2, nz-2):
            # x方向四阶差分
            dvx_dx = (c1*(vx[i+1,j] - vx[i,j]) + 
                     c2*(vx[i+2,j] - vx[i-1,j])) / dx
            
            # z方向四阶差分
            dvz_dz = (c1*(vz[i,j+1] - vz[i,j]) + 
                     c2*(vz[i,j+2] - vz[i,j-1])) / dz
            
            dvx_dz = (c1*(vx[i,j+1] - vx[i,j]) + 
                     c2*(vx[i,j+2] - vx[i,j-1])) / dz
            
            dvz_dx = (c1*(vz[i+1,j] - vz[i,j]) + 
                     c2*(vz[i+2,j] - vz[i-1,j])) / dx
            
            # 更新应力
            sigxx[i,j] = sigxx_prev[i,j] + dt * (
                c11[i,j] * dvx_dx + c13[i,j] * dvz_dz
            )
            sigzz[i,j] = sigzz_prev[i,j] + dt * (
                c13[i,j] * dvx_dx + c33[i,j] * dvz_dz
            )
            sigxz[i,j] = sigxz_prev[i,j] + dt * c44[i,j] * (
                dvx_dz + dvz_dx
            )
    
    # 3. 处理边界条件 (需要特殊处理，因为四阶差分需要更多的边界点)
    for i in range(2):
        vx[i,:] = vx[-i-1,:] = 0
        vz[i,:] = vz[-i-1,:] = 0
        sigxx[i,:] = sigxx[-i-1,:] = 0
        sigzz[i,:] = sigzz[-i-1,:] = 0
        sigxz[i,:] = sigxz[-i-1,:] = 0
        
        vx[:,i] = vx[:,-i-1] = 0
        vz[:,i] = vz[:,-i-1] = 0
        sigxx[:,i] = sigxx[:,-i-1] = 0
        sigzz[:,i] = sigzz[:,-i-1] = 0
        sigxz[:,i] = sigxz[:,-i-1] = 0

def load_model_parameters(filename, nx, nz, input_type):
    """
    读取模型参数
    
    Parameters:
        filename: 模型文件名
        nx, nz: 网格大小
        input_type: 1 - Thomsen参数 (vp,vs,rho,delta,epsilon)
                   2 - Cij参数 (c11,c13,c33,c44,rho)
    
    Returns:
        如果input_type=1: 返回c11,c13,c33,c44,rho
        如果input_type=2: 直接返回c11,c13,c33,c44,rho
    """
    try:
        # 读取5*nz*nx的数据
        data = np.loadtxt(filename)
        if data.size != 5*nz*nx:
            raise ValueError(f"Expected {5*nz*nx} values, but got {data.size}")
        
        # 重塑数据为(5,nx,nz)
        data = data.reshape(5, nz, nx).transpose(0, 2, 1)
        
        if input_type == 1:
            # 数据顺序: vp(m/s), vs(m/s), rho(kg/m³), delta, epsilon
            vp = data[0]      # m/s
            vs = data[1]      # m/s
            rho = data[2]     # kg/m³
            delta = data[3]   # 无量纲
            epsilon = data[4]  # 无量纲
            
            # 转换为Cij参数 (Pa)
            c11, c13, c33, c44 = compute_cij_from_Thomsen(vp, vs, rho, epsilon, delta)
            
        elif input_type == 2:
            # 数据顺序: c11(Pa), c13(Pa), c33(Pa), c44(Pa), rho(kg/m³)
            c11 = data[0]     # Pa
            c13 = data[1]     # Pa
            c33 = data[2]     # Pa
            c44 = data[3]     # Pa
            rho = data[4]     # kg/m³
            
        else:
            raise ValueError("input_type must be 1 or 2")
            
        return c11, c13, c33, c44, rho
        
    except Exception as e:
        print(f"Error reading model file {filename}: {str(e)}")
        raise

def forward_modeling_with_receivers(c11,c13,c33,c44,rho,
                                    nx,nz,dx,dz,dt,nt,wavelet,sx,sz,rx,rz,damping,
                                    n_sources,nt_wav, store_fields=False):
    vx=np.zeros((nx,nz))
    vz=np.zeros((nx,nz))
    sigxx=np.zeros((nx,nz))
    sigzz=np.zeros((nx,nz))
    sigxz=np.zeros((nx,nz))

    vx_prev=vx.copy()
    vz_prev=vz.copy()
    sigxx_prev=sigxx.copy()
    sigzz_prev=sigzz.copy()
    sigxz_prev=sigxz.copy()

    n_receivers = len(rx)
    synthetic_data = np.zeros((n_sources, n_receivers, nt))

    ux_hist = []
    uz_hist = []

    for it in range(nt):
        if it < nt_wav:
            for s_i in range(n_sources):
                sigxx[sx[s_i], sz[s_i]] += wavelet[it]
                sigzz[sx[s_i], sz[s_i]] += wavelet[it]
                
                # 记录当前炮点对应的所有检波器的记录
                synthetic_data[s_i, :, it] = sigxx[rx, rz]

        ux_hist.append(vx.copy())
        uz_hist.append(vz.copy())

        fd_update_staggered(vx,vz,sigxx,sigzz,sigxz,
                            vx_prev,vz_prev,sigxx_prev,sigzz_prev,sigxz_prev,
                            c11,c13,c33,c44,rho,dt,dx,dz,nx,nz)

        apply_pml(vx,damping,nx,nz)
        apply_pml(vz,damping,nx,nz)
        apply_pml(sigxx,damping,nx,nz)
        apply_pml(sigzz,damping,nx,nz)
        apply_pml(sigxz,damping,nx,nz)

        vx_prev,vx = vx,vx_prev
        vz_prev,vz = vz,vz_prev
        sigxx_prev,sigxx = sigxx,sigxx_prev
        sigzz_prev,sigzz = sigzz,sigzz_prev
        sigxz_prev,sigxz = sigxz,sigxz_prev

    ux_hist = np.array(ux_hist) # (nt,nx,nz)
    uz_hist = np.array(uz_hist)

    # 计算dtt_ux, dtt_uz
    dtt_ux_all = np.zeros((nt,nx,nz))
    dtt_uz_all = np.zeros((nt,nx,nz))
    for it in range(1, nt-1):
        dtt_ux_all[it] = (ux_hist[it+1]-2*ux_hist[it]+ux_hist[it-1])/(dt**2)
        dtt_uz_all[it] = (uz_hist[it+1]-2*uz_hist[it]+uz_hist[it-1])/(dt**2)

    if store_fields:
        forward_fields = []
        for it in range(nt):
            ux = ux_hist[it]
            uz = uz_hist[it]
            dx_ux = np.zeros((nx,nz))
            dz_ux = np.zeros((nx,nz))
            dx_uz = np.zeros((nx,nz))
            dz_uz = np.zeros((nx,nz))

            dx_ux[1:,:] = (ux[1:,:]-ux[:-1,:])/dx
            dz_ux[:,1:] = (ux[:,1:]-ux[:,:-1])/dz
            dx_uz[1:,:] = (uz[1:,:]-uz[:-1,:])/dx
            dz_uz[:,1:] = (uz[:,1:]-uz[:,:-1])/dz

            forward_fields.append({
                'ux': ux,
                'uz': uz,
                'dx_ux': dx_ux,
                'dz_ux': dz_ux,
                'dx_uz': dx_uz,
                'dz_uz': dz_uz,
                'dtt_ux': dtt_ux_all[it],
                'dtt_uz': dtt_uz_all[it],
                'rho': rho
            })
    else:
        forward_fields = None

    return synthetic_data, forward_fields

def adjoint_modeling(c11,c13,c33,c44,rho,
                     nx,nz,dx,dz,dt,nt,residual,rx,rz,damping,
                     n_sources,store_fields=False):
    """
    residual: shape (n_sources, n_receivers, nt)
    """
    adjoint_fields_all = []
    
    # 对每个炮点进行伴随波场计算
    for s_i in range(n_sources):
        vx=np.zeros((nx,nz))
        vz=np.zeros((nx,nz))
        sigxx=np.zeros((nx,nz))
        sigzz=np.zeros((nx,nz))
        sigxz=np.zeros((nx,nz))

        vx_prev=vx.copy()
        vz_prev=vz.copy()
        sigxx_prev=sigxx.copy()
        sigzz_prev=sigzz.copy()
        sigxz_prev=sigxz.copy()

        n_receivers = residual.shape[1]
        ux_hist = []
        uz_hist = []

        # 使用当前炮点的残差
        current_residual = residual[s_i]

        for it in range(nt-1, -1, -1):
            for r_i in range(n_receivers):
                sigxx[rx[r_i], rz[r_i]] += current_residual[r_i, it]
                sigzz[rx[r_i], rz[r_i]] += current_residual[r_i, it]

            ux_hist.append(vx.copy())
            uz_hist.append(vz.copy())

            fd_update_staggered(vx,vz,sigxx,sigzz,sigxz,
                                vx_prev,vz_prev,sigxx_prev,sigzz_prev,sigxz_prev,
                                c11,c13,c33,c44,rho,dt,dx,dz,nx,nz)

            apply_pml(vx,damping,nx,nz)
            apply_pml(vz,damping,nx,nz)
            apply_pml(sigxx,damping,nx,nz)
            apply_pml(sigzz,damping,nx,nz)
            apply_pml(sigxz,damping,nx,nz)

            vx_prev,vx = vx,vx_prev
            vz_prev,vz = vz,vz_prev
            sigxx_prev,sigxx = sigxx,sigxx_prev
            sigzz_prev,sigzz = sigzz,sigzz_prev
            sigxz_prev,sigxz = sigxz,sigxz_prev

        ux_hist.reverse()
        uz_hist.reverse()
        ux_hist = np.array(ux_hist)
        uz_hist = np.array(uz_hist)

        if store_fields:
            adjoint_fields = []
            for it in range(nt):
                ux_adj = ux_hist[it]
                uz_adj = uz_hist[it]
                dx_ux_adj = np.zeros((nx,nz))
                dz_ux_adj = np.zeros((nx,nz))
                dx_uz_adj = np.zeros((nx,nz))
                dz_uz_adj = np.zeros((nx,nz))

                dx_ux_adj[1:,:] = (ux_adj[1:,:]-ux_adj[:-1,:])/dx
                dz_ux_adj[:,1:] = (ux_adj[:,1:]-ux_adj[:,:-1])/dz
                dx_uz_adj[1:,:] = (uz_adj[1:,:]-uz_adj[:-1,:])/dx
                dz_uz_adj[:,1:] = (uz_adj[:,1:]-uz_adj[:,:-1])/dz

                adjoint_fields.append({
                    'ux_adj': ux_adj,
                    'uz_adj': uz_adj,
                    'dx_ux_adj': dx_ux_adj,
                    'dz_ux_adj': dz_ux_adj,
                    'dx_uz_adj': dx_uz_adj,
                    'dz_uz_adj': dz_uz_adj
                })
            adjoint_fields_all.append(adjoint_fields)
        else:
            adjoint_fields_all.append(None)

    return adjoint_fields_all

def compute_gradient(forward_fields, adjoint_fields, nx, nz, nt):
    grad_c11 = np.zeros((nx,nz))
    grad_c13 = np.zeros((nx,nz))
    grad_c33 = np.zeros((nx,nz))
    grad_c44 = np.zeros((nx,nz))
    grad_rho = np.zeros((nx,nz))

    rho = forward_fields[0]['rho']

    for it in range(nt):
        dx_ux = forward_fields[it]['dx_ux']
        dz_ux = forward_fields[it]['dz_ux']
        dx_uz = forward_fields[it]['dx_uz']
        dz_uz = forward_fields[it]['dz_uz']
        dtt_ux = forward_fields[it]['dtt_ux']
        dtt_uz = forward_fields[it]['dtt_uz']
        ux = forward_fields[it]['ux']
        uz = forward_fields[it]['uz']

        dx_ux_adj = adjoint_fields[it]['dx_ux_adj']
        dz_ux_adj = adjoint_fields[it]['dz_ux_adj']
        dx_uz_adj = adjoint_fields[it]['dx_uz_adj']
        dz_uz_adj = adjoint_fields[it]['dz_uz_adj']
        ux_adj = adjoint_fields[it]['ux_adj']
        uz_adj = adjoint_fields[it]['uz_adj']

        grad_c11 += -(dx_ux_adj * dx_ux)

    
        term_c13 = (dz_uz_adj * dx_ux) + (dx_ux_adj * dz_uz) + (dz_uz_adj * dx_ux) + (dx_ux_adj * dz_uz)
        grad_c13 += -term_c13

        grad_c33 += -(dz_uz_adj * dz_uz)

        adj_sum = dz_ux_adj + dx_uz_adj
        fwd_sum = dz_ux + dx_uz
        grad_c44 += -(adj_sum * fwd_sum)

        grad_rho += -(ux_adj * (rho * dtt_ux) + uz_adj * (rho * dtt_uz))

    return grad_c11, grad_c13, grad_c33, grad_c44, grad_rho

def save_gradient(grad_c11, grad_c13, grad_c33, grad_c44, grad_rho, iteration):
    np.savetxt(f"grad_c11_iter{iteration}.txt", grad_c11)
    np.savetxt(f"grad_c13_iter{iteration}.txt", grad_c13)
    np.savetxt(f"grad_c33_iter{iteration}.txt", grad_c33)
    np.savetxt(f"grad_c44_iter{iteration}.txt", grad_c44)
    np.savetxt(f"grad_rho_iter{iteration}.txt", grad_rho)

def parabola_line_search(misfit_old, misfit_mid, misfit_new, a_old, a_mid, a_new):
    M = np.array([[a_old**2, a_old, 1],
                  [a_mid**2, a_mid, 1],
                  [a_new**2, a_new, 1]])
    y = np.array([misfit_old, misfit_mid, misfit_new])
    coeff = np.linalg.lstsq(M,y,rcond=None)[0] # A,B,C
    A,B,C = coeff
    if A==0:
        return a_mid
    a_opt = -B/(2*A)
    if a_opt<=0:
        a_opt = a_mid
    return a_opt

def compute_Thomsen_from_cij(c11, c13, c33, c44, rho):
    # 添加参数有效性检查
    if np.any(c33 <= 0) or np.any(rho <= 0):
        raise ValueError("c33 and rho must be positive")
    if np.any(c44 <= 0):
        raise ValueError("c44 must be positive")
    
    # 计算垂直方向的P波和S波速度
    vp = np.sqrt(c33/rho)  # 垂直P波速度
    vs = np.sqrt(c44/rho)  # 垂直S波速度
    
    # 计算Thomsen参数
    epsilon = (c11 - c33)/(2*c33)
    
    # 计算delta参数
    term = (c13 + c44)**2 - (c33 - c44)**2
    delta = term/(2*c33*(c33 - c44))
    
    return vp, vs, rho, epsilon, delta

# 在main函数的迭代结束后添加:
def save_final_model(c11, c13, c33, c44, rho, iteration):
    """
    保存最终的模型参数
    """
    # 转换为Thomsen参数
    vp, vs, rho, epsilon, delta = compute_Thomsen_from_cij(c11, c13, c33, c44, rho)
    
    # 保存结果
    np.savetxt(f"final_vp_iter{iteration}.txt", vp)
    np.savetxt(f"final_vs_iter{iteration}.txt", vs)
    np.savetxt(f"final_rho_iter{iteration}.txt", rho)
    np.savetxt(f"final_epsilon_iter{iteration}.txt", epsilon)
    np.savetxt(f"final_delta_iter{iteration}.txt", delta)
    
    print("Final model parameters saved:")
    print(f"Average Vp: {np.mean(vp):.2f} m/s")
    print(f"Average Vs: {np.mean(vs):.2f} m/s")
    print(f"Average density: {np.mean(rho):.2f} kg/m³")
    print(f"Average epsilon: {np.mean(epsilon):.4f}")
    print(f"Average delta: {np.mean(delta):.4f}")

def plot_models(model_type, nx, nz, dx, dz,
               plot_true_model=True, plot_init_model=True,
               vp_init=None, vs_init=None, rho_init=None, delta_init=None, epsilon_init=None,
               vp_true=None, vs_true=None, rho_true=None, delta_true=None, epsilon_true=None,
               c11_init=None, c13_init=None, c33_init=None, c44_init=None,
               c11_true=None, c13_true=None, c33_true=None, c44_true=None):
    """
    绘制模型参数对比图
    
    Parameters:
        model_type: 1 - Thomsen参数 (vp,vs,rho,delta,epsilon)
                   2 - Cij参数 (c11,c13,c33,c44,rho)
        nx, nz: 网格点数
        dx, dz: 网格间距
        plot_true_model: 是否绘制真实模型
        plot_init_model: 是否绘制初始模型
    """
    import matplotlib.pyplot as plt
    
    # 创建坐标网格
    x = np.arange(nx) * dx / 1000  # 转换为km
    z = np.arange(nz) * dz / 1000
    X, Z = np.meshgrid(x, z)
    
    if model_type == 1:  # Thomsen参数
        fig, axes = plt.subplots(5, 2, figsize=(12, 20))
        fig.suptitle('Thomsen Parameters Comparison', fontsize=16)
        
        # Vp (m/s)
        if plot_init_model and vp_init is not None:
            im1 = axes[0,0].pcolor(X, Z, vp_init.T, shading='auto')
            axes[0,0].set_title('Initial Vp (m/s)')
            plt.colorbar(im1, ax=axes[0,0])
        if plot_true_model and vp_true is not None:
            im2 = axes[0,1].pcolor(X, Z, vp_true.T, shading='auto')
            axes[0,1].set_title('True Vp (m/s)')
            plt.colorbar(im2, ax=axes[0,1])
            
        # Vs (m/s)
        if plot_init_model and vs_init is not None:
            im3 = axes[1,0].pcolor(X, Z, vs_init.T, shading='auto')
            axes[1,0].set_title('Initial Vs (m/s)')
            plt.colorbar(im3, ax=axes[1,0])
        if plot_true_model and vs_true is not None:
            im4 = axes[1,1].pcolor(X, Z, vs_true.T, shading='auto')
            axes[1,1].set_title('True Vs (m/s)')
            plt.colorbar(im4, ax=axes[1,1])
            
        # Density (kg/m³)
        if plot_init_model and rho_init is not None:
            im5 = axes[2,0].pcolor(X, Z, rho_init.T, shading='auto')
            axes[2,0].set_title('Initial Density (kg/m³)')
            plt.colorbar(im5, ax=axes[2,0])
        if plot_true_model and rho_true is not None:
            im6 = axes[2,1].pcolor(X, Z, rho_true.T, shading='auto')
            axes[2,1].set_title('True Density (kg/m³)')
            plt.colorbar(im6, ax=axes[2,1])
            
        # Delta
        if plot_init_model and delta_init is not None:
            im7 = axes[3,0].pcolor(X, Z, delta_init.T, shading='auto')
            axes[3,0].set_title('Initial Delta')
            plt.colorbar(im7, ax=axes[3,0])
        if plot_true_model and delta_true is not None:
            im8 = axes[3,1].pcolor(X, Z, delta_true.T, shading='auto')
            axes[3,1].set_title('True Delta')
            plt.colorbar(im8, ax=axes[3,1])
            
        # Epsilon
        if plot_init_model and epsilon_init is not None:
            im9 = axes[4,0].pcolor(X, Z, epsilon_init.T, shading='auto')
            axes[4,0].set_title('Initial Epsilon')
            plt.colorbar(im9, ax=axes[4,0])
        if plot_true_model and epsilon_true is not None:
            im10 = axes[4,1].pcolor(X, Z, epsilon_true.T, shading='auto')
            axes[4,1].set_title('True Epsilon')
            plt.colorbar(im10, ax=axes[4,1])
            
    elif model_type == 2:  # Cij参数
        fig, axes = plt.subplots(4, 2, figsize=(12, 16))
        fig.suptitle('Elastic Parameters Comparison (GPa)', fontsize=16)
        
        # C11
        if plot_init_model and c11_init is not None:
            im1 = axes[0,0].pcolor(X, Z, c11_init.T/1e9, shading='auto')
            axes[0,0].set_title('Initial C11')
            plt.colorbar(im1, ax=axes[0,0])
        if plot_true_model and c11_true is not None:
            im2 = axes[0,1].pcolor(X, Z, c11_true.T/1e9, shading='auto')
            axes[0,1].set_title('True C11')
            plt.colorbar(im2, ax=axes[0,1])
            
        # C13
        if plot_init_model and c13_init is not None:
            im3 = axes[1,0].pcolor(X, Z, c13_init.T/1e9, shading='auto')
            axes[1,0].set_title('Initial C13')
            plt.colorbar(im3, ax=axes[1,0])
        if plot_true_model and c13_true is not None:
            im4 = axes[1,1].pcolor(X, Z, c13_true.T/1e9, shading='auto')
            axes[1,1].set_title('True C13')
            plt.colorbar(im4, ax=axes[1,1])
            
        # C33
        if plot_init_model and c33_init is not None:
            im5 = axes[2,0].pcolor(X, Z, c33_init.T/1e9, shading='auto')
            axes[2,0].set_title('Initial C33')
            plt.colorbar(im5, ax=axes[2,0])
        if plot_true_model and c33_true is not None:
            im6 = axes[2,1].pcolor(X, Z, c33_true.T/1e9, shading='auto')
            axes[2,1].set_title('True C33')
            plt.colorbar(im6, ax=axes[2,1])
            
        # C44
        if plot_init_model and c44_init is not None:
            im7 = axes[3,0].pcolor(X, Z, c44_init.T/1e9, shading='auto')
            axes[3,0].set_title('Initial C44')
            plt.colorbar(im7, ax=axes[3,0])
        if plot_true_model and c44_true is not None:
            im8 = axes[3,1].pcolor(X, Z, c44_true.T/1e9, shading='auto')
            axes[3,1].set_title('True C44')
            plt.colorbar(im8, ax=axes[3,1])
    
    # 设置所有子图的坐标轴标签
    for ax in axes.flat:
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Depth (km)')
    
    plt.tight_layout()
    plt.show()

def plot_wavelet(wavelet, dt, t0, f0):  # 添加f0参数
    """
    绘制雷克子波
    
    Parameters:
        wavelet: 子波数据
        dt: 采样间隔(s)
        t0: 时间偏移(s)
        f0: 主频(Hz)
    """
    import matplotlib.pyplot as plt
    
    nt = len(wavelet)
    t = np.linspace(0, (nt-1)*dt, nt) + t0
    
    plt.figure(figsize=(10, 4))
    plt.plot(t, wavelet, 'b-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Ricker Wavelet')
    
    # 添加主要参数信息
    info_text = f'dt = {dt*1000:.1f} ms\nt0 = {t0*1000:.1f} ms\nf0 = {f0:.1f} Hz'
    plt.text(0.02, 0.98, info_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

@jit(nopython=True)
def check_stability_condition(c11, c33, c44, rho, dx, dz, dt):
    """
    检查四阶差分的CFL稳定性条件
    
    对于四阶差分，CFL条件需要更严格
    """
    # 计算最大波速
    vp_h = np.sqrt(np.max(c11/rho))  # 水平P波速度
    vp_v = np.sqrt(np.max(c33/rho))  # 垂直P波速度
    vs = np.sqrt(np.max(c44/rho))    # S波速度
    
    v_max = max(vp_h, vp_v, vs)      # 取最大速度
    
    # 四阶差分的修正系数（比二阶差分更严格）
    # 参考：Levander, A. R. (1988). Fourth-order finite-difference P-SV seismograms
    coef = 0.3  # 四阶差分的稳定性系数
    
    # 计算CFL数
    cfl = v_max * dt * np.sqrt(1/dx**2 + 1/dz**2)
    
    return cfl <= coef

def analyze_dispersion(dx, dz, f0, vp, vs):
    """
    分析数值分散性
    
    Args:
        dx, dz: 网格间距
        f0: 主频率
        vp, vs: P波和S波速度
    
    Returns:
        float: 每个波长的网格点数
    """
    # 计算最小波长
    v_min = min(vp, vs)
    lambda_min = v_min / f0
    
    # 计算每个波长的网格点数
    points_per_wavelength = lambda_min / max(dx, dz)
    
    return points_per_wavelength

def main():
    #-------------------------------------------------------------------
    #                       模型参数设置
    #-------------------------------------------------------------------
    # 网格参数
    nx, nz = 500, 500          # 网格点数
    dx, dz = 10.0, 10.0        # 网格间距(m)
    
    # 时间参数
    dt = 0.001                 # 时间步长(s)
    nt = 4001                  # 总时间步数
    
    # PML参数
    pml_size = 50              # PML层厚度(网格点数)
    damping_coeff = 0.015      # PML衰减系数
    
    # 震源参数
    f0 = 25.0                  # 雷克子波主频(Hz)
    t0 = 0.1                   # 子波时间偏移(s)
    nt_wav = 100               # 震源时间函数长度(采样点数)
    
    # 震源布置
    source_x_start = 200       # 第一个震源x坐标(网格点)
    source_z_start = 2       # 第一个震源z坐标(网格点)
    dx_s, dz_s = 100, 0        # 震源间隔(网格点)
    n_sources = 3              # 震源个数
    
    # 检波器布置
    rec_x_start = 50          # 第一个检波器x坐标(网格点)
    rec_z_start = 5         # 第一个检波器z坐标(网格点)
    dx_r, dz_r = 5, 0        # 检波器间隔(网格点)
    n_receivers = 81         # 检波器个数
    
    # 反演参数
    n_iterations = 10         # 最大迭代次数
    a_current = 1e-4         # 初始步长
    save_interval = 10       # 中间结果保存间隔
    
    # 模型文件设置
    input_type = 1    # 1: Thomsen参数, 2: Cij参数
    true_model = "emodel.xyz"    # 真实模型文件
    init_model = "emodel.xyz"    # 初始模型文件
    
    # 绘图控制
    plot_true_model = True     # 是否绘制真实模型
    plot_init_model = True     # 是否绘制初始模型
    plot_final_model = True    # 是否绘制最终反演结果
    plot_source_wavelet = True # 是否绘制震源子波
    
    #-------------------------------------------------------------------
    #                       模型加载
    #-------------------------------------------------------------------
    print("Loading initial model...")
    try:
        c11_init, c13_init, c33_init, c44_init, rho_init = load_model_parameters(
            init_model, nx, nz, input_type)
    except Exception as e:
        print(f"Error loading initial model: {str(e)}")
        return
        
    print("Loading true model...")
    try:
        c11_real, c13_real, c33_real, c44_real, rho_real = load_model_parameters(
            true_model, nx, nz, input_type)
    except Exception as e:
        print(f"Error loading true model: {str(e)}")
        return
    
    # 加载模型后添加初始模型对比图
    if plot_true_model or plot_init_model:
        print("Plotting initial and true models...")
        if input_type == 1:
            # 如果输入是Thomsen参数，先转换回Thomsen参数再绘图
            vp_init, vs_init, rho_init, epsilon_init, delta_init = compute_Thomsen_from_cij(
                c11_init, c13_init, c33_init, c44_init, rho_init)
            vp_true, vs_true, rho_true, epsilon_true, delta_true = compute_Thomsen_from_cij(
                c11_real, c13_real, c33_real, c44_real, rho_real)
            
            plot_models(1, nx, nz, dx, dz,
                       plot_true_model, plot_init_model,
                       vp_init, vs_init, rho_init, delta_init, epsilon_init,
                       vp_true, vs_true, rho_true, delta_true, epsilon_true)
        else:
            plot_models(2, nx, nz, dx, dz,
                       plot_true_model, plot_init_model,
                       c11_init=c11_init, c13_init=c13_init, c33_init=c33_init, c44_init=c44_init,
                       c11_true=c11_real, c13_true=c13_real, c33_true=c33_real, c44_true=c44_real)
    
    #-------------------------------------------------------------------
    #                       主程序开始
    #-------------------------------------------------------------------
    
    # 生成震源和检波器位置
    sx, sz = generate_positions(source_x_start, source_z_start, dx_s, dz_s, n_sources)
    rx, rz = generate_positions(rec_x_start, rec_z_start, dx_r, dz_r, n_receivers)
    
    # 初始化PML
    damping = initialize_pml(nx, nz, pml_size, damping_coeff)
    
    # 生成带时移的雷克子波
    wavelet = ricker_wavelet(f0, dt, nt_wav, t0)
    
    # 绘制震源子波
    if plot_source_wavelet:
        print("Plotting source wavelet...")
        plot_wavelet(wavelet, dt, t0, f0)
    
    # 使用真实模型生成观测数据
    observed_data, _ = forward_modeling_with_receivers(c11_real,c13_real,c33_real,c44_real,rho_real,
                                                       nx,nz,dx,dz,dt,nt,wavelet,sx,sz,rx,rz,damping,
                                                       n_sources,nt_wav,store_fields=False)
    
    # 初始化反演参数
    c11 = c11_init.copy()
    c13 = c13_init.copy()
    c33 = c33_init.copy()
    c44 = c44_init.copy()
    rho = rho_init.copy()

    synthetic_data, forward_fields = forward_modeling_with_receivers(c11,c13,c33,c44,rho,
                                                                     nx,nz,dx,dz,dt,nt,wavelet,sx,sz,rx,rz,damping,
                                                                     n_sources,nt_wav,store_fields=True)
    residual = synthetic_data - observed_data
    initial_misfit = 0.5*np.sum(residual**2)
    print("Initial misfit:", initial_misfit)

    #-------------------------------------------------------------------
    #                       反演迭代
    #-------------------------------------------------------------------
    for iteration in range(1, n_iterations+1):
        # 求伴随场
        adjoint_fields = adjoint_modeling(c11,c13,c33,c44,rho,
                                          nx,nz,dx,dz,dt,nt,residual,rx,rz,damping,
                                          store_fields=True)

        # 计算梯度
        grad_c11, grad_c13, grad_c33, grad_c44, grad_rho = compute_gradient(forward_fields, adjoint_fields, nx, nz, nt)

        # 保存梯度
        save_gradient(grad_c11, grad_c13, grad_c33, grad_c44, grad_rho, iteration)

        # 抛物线法线搜索
        a_old = a_current/2
        a_new = a_current*2
        misfit_mid = 0.5*np.sum((synthetic_data - observed_data)**2)

        # 测试a_old
        c11_test = c11 - a_old*grad_c11
        c13_test = c13 - a_old*grad_c13
        c33_test = c33 - a_old*grad_c33
        c44_test = c44 - a_old*grad_c44
        rho_test = rho - a_old*grad_rho
        syn_test, _ = forward_modeling_with_receivers(c11_test,c13_test,c33_test,c44_test,rho_test,
                                                      nx,nz,dx,dz,dt,nt,wavelet,sx,sz,rx,rz,damping,
                                                      n_sources,nt_wav,store_fields=False)
        misfit_old = 0.5*np.sum((syn_test - observed_data)**2)

        # 测试a_new
        c11_test2 = c11 - a_new*grad_c11
        c13_test2 = c13 - a_new*grad_c13
        c33_test2 = c33 - a_new*grad_c33
        c44_test2 = c44 - a_new*grad_c44
        rho_test2 = rho - a_new*grad_rho
        syn_test2,_ = forward_modeling_with_receivers(c11_test2,c13_test2,c33_test2,c44_test2,rho_test2,
                                                      nx,nz,dx,dz,dt,nt,wavelet,sx,sz,rx,rz,damping,
                                                      n_sources,nt_wav,store_fields=False)
        misfit_new = 0.5*np.sum((syn_test2 - observed_data)**2)

        a_current = parabola_line_search(misfit_old, misfit_mid, misfit_new, a_old, a_current, a_new)

        # 使用a_current更新模型
        c11 -= a_current*grad_c11
        c13 -= a_current*grad_c13
        c33 -= a_current*grad_c33
        c44 -= a_current*grad_c44
        rho -= a_current*grad_rho

        # 新的正演计算新的misfit
        synthetic_data, forward_fields = forward_modeling_with_receivers(c11,c13,c33,c44,rho,
                                                                         nx,nz,dx,dz,dt,nt,wavelet,sx,sz,rx,rz,damping,
                                                                         n_sources,nt_wav,store_fields=True)
        residual = synthetic_data - observed_data
        misfit = 0.5*np.sum(residual**2)

        print(f"Iteration {iteration}, Misfit: {misfit}, step: {a_current}")

        if misfit < 0.01*initial_misfit:
            print("Misfit below 1% of initial, stopping.")
            break

        # 每隔save_interval次迭代保存一次结果
        if iteration % save_interval == 0:
            print(f"Saving intermediate results at iteration {iteration}")
            save_final_model(c11, c13, c33, c44, rho, iteration)
            
    # 保存最终模型
    save_final_model(c11, c13, c33, c44, rho, iteration)
    print("FWI completed. Final misfit:", misfit)
    print("Model parameters updated.")

    # 迭代结束后绘制最终结果
    if plot_final_model:
        print("\nPlotting final inversion results...")
        # 将最终的Cij结果转换为Thomsen参数
        vp_final, vs_final, rho_final, epsilon_final, delta_final = compute_Thomsen_from_cij(
            c11, c13, c33, c44, rho)
            
        # 绘制最终结果与真实模型的对比
        plot_models(1, nx, nz, dx, dz,
                   True, True,  # 同时显示真实模型和反演结果
                   vp_final, vs_final, rho_final, delta_final, epsilon_final,  # 反演结果
                   vp_true, vs_true, rho_true, delta_true, epsilon_true)      # 真实模型
        
        # 打印最终结果的统计信息
        print("\nFinal model parameters:")
        print(f"Average Vp: {np.mean(vp_final):.2f} m/s")
        print(f"Average Vs: {np.mean(vs_final):.2f} m/s")
        print(f"Average density: {np.mean(rho_final):.2f} kg/m³")
        print(f"Average epsilon: {np.mean(epsilon_final):.4f}")
        print(f"Average delta: {np.mean(delta_final):.4f}")
        
        # 计算相对误差
        print("\nRelative errors:")
        print(f"Vp error: {np.mean(np.abs((vp_final-vp_true)/vp_true))*100:.2f}%")
        print(f"Vs error: {np.mean(np.abs((vs_final-vs_true)/vs_true))*100:.2f}%")
        print(f"Density error: {np.mean(np.abs((rho_final-rho_true)/rho_true))*100:.2f}%")
        print(f"Epsilon error: {np.mean(np.abs(epsilon_final-epsilon_true)):.4f}")
        print(f"Delta error: {np.mean(np.abs(delta_final-delta_true)):.4f}")

if __name__ == "__main__":
    main()
