# 声波各向同性FWI项目文档

## 项目概述
这是一个基于MATLAB实现的声波各向同性(Acoustic Isotropic)媒质中的全波形反演(FWI)框架。该项目采用函数式编程范式，通过独立的函数模块实现波场正演模拟、残差计算、伴随波场反传以及梯度计算等功能，用于地震勘探领域的精确地下结构成像。

## 核心功能
1. **波场模拟**
   - 实现了声波介质中的波场传播模拟
   - 支持自由表面和PML吸收边界条件
   - 基于一阶声波方程系统的显式时间积分
   - 提供完整波场存储和检波器记录输出

2. **全波形反演(FWI)**
   - 完整的FWI工作流程实现
   - 基于伴随状态方法的梯度计算
   - 提供多种目标函数(波形残差、归一化互相关等)
   - 使用共轭梯度法进行模型参数优化

## 项目结构及核心组件详解

### `external/fd_p2D_fs.m`
声波方程的二维有限差分正演模拟函数，支持自由表面条件。

#### 函数定义
```matlab
function [shot_data, wavefield] = fd_p2D_fs(vp, dx, dt, npml, srcx, srcz, nrec, recx, recz, nt, nx, nz, f0, wavelet, snap_interval)
```

#### 参数详解
- **模型参数**:
  - `vp`: 纵波速度模型，二维数组(nx, nz)，单位为米/秒(m/s)
  - `dx`: 空间采样间隔，单位为米(m)
  - `dt`: 时间采样间隔，单位为秒(s)
  
- **边界参数**:
  - `npml`: PML区域的厚度，单位为网格点数
  
- **震源参数**:
  - `srcx`, `srcz`: 震源的x和z坐标(从1开始的索引)
  - `f0`: 震源的主频率，单位为赫兹(Hz)
  - `wavelet`: 震源子波，如果为空则内部生成Ricker子波
  
- **检波器参数**:
  - `nrec`: 检波器数量
  - `recx`, `recz`: 检波器的x和z坐标数组(从1开始的索引)
  
- **计算参数**:
  - `nt`: 时间步数
  - `nx`, `nz`: 模型在x和z方向的网格点数
  - `snap_interval`: 波场快照保存间隔(时间步)

#### 函数流程
1. 初始化波场变量(压力p、质点速度vx和vz)
2. 设置PML吸收边界参数
3. 生成或使用提供的震源子波
4. 进行时间步进计算:
   - 更新质点速度
   - 应用PML边界条件
   - 更新压力场
   - 添加震源项
   - 应用自由表面条件
   - 记录检波器数据
   - 保存波场快照(如果需要)

#### 边界条件
- 使用标准PML(Perfectly Matched Layer)吸收边界
- 模型顶部采用自由表面条件(压力为零)

#### 输出
- `shot_data`: 检波器记录的压力数据，二维数组(nt, nrec)
- `wavefield`: 如果要求保存波场，则包含所选时间步的完整波场快照

#### 调用示例
```matlab
% 设置模型和计算参数
vp = ones(101, 101) * 2000; % 2000 m/s均匀模型
dx = 10; % 10 m空间采样
dt = 0.001; % 1 ms时间采样
npml = 20; % 20个网格点的PML厚度
srcx = 51; srcz = 1; % 震源位置
nrec = 99; % 99个检波器
recx = 2:100; recz = ones(1, 99); % 检波器位置
nt = 1000; % 1000个时间步
nx = 101; nz = 101; % 模型大小
f0 = 15; % 15 Hz震源频率
wavelet = []; % 使用内部生成的Ricker子波
snap_interval = 10; % 每10个时间步保存一次波场快照

% 调用正演函数
[shot_data, wavefield] = fd_p2D_fs(vp, dx, dt, npml, srcx, srcz, nrec, recx, recz, nt, nx, nz, f0, wavelet, snap_interval);
```

### `external/fd_p2D_fs_re.m`
声波方程的二维有限差分反传函数，用于计算伴随波场，支持自由表面条件。

#### 函数定义
```matlab
function [shot_data, wavefield] = fd_p2D_fs_re(vp, dx, dt, npml, recx, recz, nrec, srcx, srcz, nt, nx, nz, residual)
```

#### 参数详解
与正演函数类似，但有以下不同:
- `residual`: 观测数据和合成数据之间的残差，用作伴随源

#### 函数流程
1. 初始化波场变量
2. 设置PML参数
3. 以时间反向顺序进行计算:
   - 在检波器位置添加残差作为伴随源
   - 更新质点速度(反向时间)
   - 应用PML边界条件
   - 更新压力场
   - 应用自由表面条件
   - 记录伴随波场

#### 特点
- 采用时间反传技术计算伴随波场
- 在检波器位置添加残差作为震源
- 支持与正演函数相同的边界条件

### `external/fd_p2D_fs_re_wav.m`
与`fd_p2D_fs_re.m`类似，但专门用于计算和保存完整的伴随波场，便于后续梯度计算。

#### 函数定义
```matlab
function [wavefield_p, wavefield_vx, wavefield_vz] = fd_p2D_fs_re_wav(vp, dx, dt, npml, recx, recz, nrec, srcx, srcz, nt, nx, nz, residual)
```

#### 特点
- 保存完整的伴随波场(压力和质点速度)
- 优化内存使用，适合大规模模型计算

### `external/adjwavest.m`
实现震源子波估计的伴随状态方法。

#### 函数定义
```matlab
function [w_adj] = adjwavest(obs, syn, vp, dx, dt, npml, srcx, srcz, nrec, recx, recz, nt, nx, nz)
```

#### 参数详解
- `obs`: 观测数据
- `syn`: 合成数据
- 其他参数与前面函数相同

#### 函数流程
1. 计算观测数据与合成数据之间的残差
2. 使用伴随方法反传残差
3. 在震源位置提取伴随波场
4. 对伴随波场进行处理得到震源子波估计

### `external/fwi_CG_simple.m`
使用共轭梯度法实现的FWI主控制函数。

#### 函数定义
```matlab
function [vp_inv, obj_func] = fwi_CG_simple(vp_init, obs_data, dx, dt, npml, srcx, srcz, nrec, recx, recz, nt, nx, nz, f0, niter)
```

#### 参数详解
- `vp_init`: 初始速度模型
- `obs_data`: 观测数据
- `niter`: 最大迭代次数
- 其他参数与前面函数相同

#### 函数流程
1. 初始化参数和变量
2. 迭代计算:
   - 使用当前模型进行正演模拟
   - 计算目标函数值
   - 计算梯度
   - 确定搜索方向(使用Fletcher-Reeves共轭梯度法)
   - 进行一维线搜索确定步长
   - 更新模型参数
   - 检查收敛条件

#### 关键技术
- Fletcher-Reeves共轭梯度法确定搜索方向
- 基于黄金分割法的一维线搜索
- 使用前处理技术提高梯度质量

#### 共轭梯度优化框架详解

##### 迭代初始化
```matlab
k = 1;                    % 当前迭代次数
kmax = 500;               % 最大迭代次数
objval = zeros(kmax,1);   % 存储每次迭代的目标函数值
```

##### 第一次迭代（最速下降方向）
```matlab
% 计算归一化的梯度方向
p = 100*grad_stk1/norm(grad_stk1);   % 归一化并放大梯度
d1 = -2*p;                           % 最速下降方向（负梯度方向）
```

##### 步长控制
```matlab
maxd = max(max(abs(d1)));            % 计算最大更新量
dec = 0.5;                           % 步长衰减因子
ks = 1;                              % 线搜索计数器

% 初始步长
a = 1;
% 控制最大更新步长不超过30
while maxd > 30
    a = dec*a;                       % 如果更新太大，减小步长
    maxd = max(max(abs(a*d1)));      % 重新计算最大更新量
end
```
这部分确保每次迭代的模型更新幅度不会太大，防止算法不稳定。

##### 线搜索过程
```matlab
while (ks < 10)                      % 最多尝试10次
    % 更新速度模型
    v0s1 = v0 + a*d1;               % 试探性更新
    v0s1 = control_v(v0s1,water,vmin,vmax);  % 应用速度约束
    
    % 计算新模型的目标函数值和梯度
    [objval1,grad_stk2] = objfun2_bx(v0s1,nz,nx,pml,dz,dx,dt,Nt,nshot,...
                                    seisture,wav2,sx,sz,rx,water);
    
    % 判断是否接受更新
    if objval1 < objval0            % 如果目标函数值减小
        v0 = v0s1;                  % 接受新模型
        objold = objval0;           % 保存旧的目标函数值
        objval0 = objval1;          % 更新目标函数值
        grad_stk1 = grad_stk2;      % 更新梯度
        p0 = p;                     % 保存旧的梯度方向
        d0 = a*d1;                  % 保存搜索方向
        break;
    else
        a = dec*a;                  % 如果目标函数值增加，减小步长
        a2 = a;                     % 保存当前步长供下次迭代使用
        ks = ks + 1;                % 增加搜索次数
    end
end
```
线搜索是一个关键步骤，通过不断尝试不同的步长，找到使目标函数值最小的更新量。

##### 后续迭代中的共轭梯度计算
```matlab
% 计算共轭梯度方向
p = 100*grad_stk1/norm(grad_stk1);    % 归一化当前梯度
% 计算共轭因子beta（Fletcher-Reeves公式）
b = sqrt(sum(sum((p'*p).^2))/sum(sum((p0'*p0).^2)));
% 计算共轭方向：当前负梯度方向 + beta*前一次方向
d1 = -p + b*d0;
```
这是共轭梯度法的核心部分，使用Fletcher-Reeves公式计算共轭因子beta，将当前梯度方向与之前的搜索方向相结合，形成新的搜索方向。

##### 完整迭代循环
```matlab
while(k < kmax)
    % 显示当前迭代次数
    k
    % 显示当前速度模型和梯度
    figure(1); imagesc(v0); pause(0.0001);
    figure(2); imagesc(grad_stk1); pause(0.0001);
    
    % 记录目标函数值
    objval(k+1) = objval0;
    
    % 计算共轭梯度方向
    p = 100*grad_stk1/norm(grad_stk1);
    b = sqrt(sum(sum((p'*p).^2))/sum(sum((p0'*p0).^2)));
    d1 = -p + b*d0;

    % 步长控制
    maxd = max(max(abs(d1)));
    dec = 0.5;
    a = 1;
    while maxd > 30
        a = dec*a;
        maxd = max(max(abs(a*d1)));
    end
    
    % 使用上一次线搜索的经验
    if ks > 1
       a = a2; 
    end
    
    % 线搜索过程
    ks = 1;
    while (ks < 10)
        v0s1 = v0 + a*d1;
        v0s1 = control_v(v0s1,water,vmin,vmax);

        [objval1,grad_stk2] = objfun2_bx(v0s1,nz,nx,pml,dz,dx,dt,Nt,nshot,...
                                        seisture,wav2,sx,sz,rx,water);
        if objval1 < objval0
            v0 = v0s1;
            objold = objval0;
            objval0 = objval1;
            grad_stk1 = grad_stk2;
            p0 = p;
            d0 = a*d1;
            break;
        else
            a = dec*a;
            a2 = a;
            ks = ks + 1;
        end
    end

    % 水层梯度置零
    grad_stk1(1:water,:) = 0;
    
    % 保存结果
    if mod(k,5) == 0
        save(['k=' num2str(k) '_v0.mat'],'v0');
        save(['k=' num2str(k) '_grad_stk1.mat'],'grad_stk1');
    end
    
    % 保存目标函数值
    save objval objval

    % 检查终止条件
    if (ks == 20)
        break;
    end
    
    % 更新迭代次数
    k = k + 1;
end
```

#### 关键算法细节

##### Fletcher-Reeves共轭梯度法
共轭梯度法是一种高效的优化算法，能够避免最速下降法中的"之字形"路径，加速收敛。Fletcher-Reeves公式计算共轭因子：
```matlab
b = sqrt(sum(sum((p'*p).^2))/sum(sum((p0'*p0).^2)));
```
这个公式计算当前梯度与上一步梯度的比值，用于确定共轭方向。

##### 自适应步长控制
代码中使用了两层步长控制：
1. 基于模型更新幅度的预先控制：
   ```matlab
   while maxd > 30
       a = dec*a;
       maxd = max(max(abs(a*d1)));
   end
   ```
2. 基于目标函数减小的线搜索：
   ```matlab
   if objval1 < objval0
       // 接受更新
   else
       a = dec*a;  // 减小步长
   ```

##### 模型约束
使用`control_v`函数对更新后的模型进行约束，确保模型参数在物理合理的范围内：
```matlab
v0s1 = control_v(v0s1,water,vmin,vmax);
```

##### 水层处理
对于海洋环境的模拟，水层的速度通常是已知的，不需要反演。代码通过将水层对应的梯度设为零来实现：
```matlab
grad_stk1(1:water,:) = 0;
```

#### 移植到VTI-Elastic的要点

1. **搜索方向计算**:
   - 保持Fletcher-Reeves公式的核心逻辑
   - 扩展到多参数情况（c11, c13, c33, c44, rho）

2. **步长控制**:
   - 需要针对弹性参数设置合适的更新幅度限制
   - 可能需要为不同参数设置不同的步长

3. **模型约束**:
   - 扩展`control_v`函数以处理多个弹性参数
   - 确保参数间的物理关系保持合理（例如c11*c33-c13^2 > 0）

4. **水层和模型边界处理**:
   - 水层处理需要扩展到多个弹性参数
   - 保持边界区域的特殊处理

5. **目标函数和梯度计算**:
   - 将`objfun2_bx`函数替换为VTI-Elastic中的目标函数计算
   - 确保梯度计算的准确性和一致性

6. **迭代控制和终止条件**:
   - 保持相似的迭代和终止逻辑
   - 可能需要调整收敛条件以适应弹性参数

### `external/cross_resid.m`和`external/cross_resid2.m`
实现基于归一化互相关的残差和目标函数计算。

#### 函数定义
```matlab
function [residual, func] = cross_resid(observed, synthetic, dt)
```

#### 特点
- 计算时间域上的归一化互相关残差
- 提供对走时偏移更敏感的目标函数
- 可减轻震源子波不匹配的影响

### `external/objfun1.m`和`external/objfun2.m`
不同类型的目标函数计算实现。

#### 函数定义
```matlab
function [func, residual] = objfun1(observed, synthetic, dt)
```

#### 特点
- `objfun1.m`: 实现基于L2范数的波形残差
- `objfun2.m`: 实现基于包络的目标函数

### 辅助函数

#### `external/taylor_vz.m`
使用泰勒展开计算速度分量，提高边界处理的精度。

#### `external/axx.m`
实现速度模型的艾克曼平滑，用于梯度预处理。

#### `external/adj_sour.m`
实现伴随源计算的辅助函数。

#### `external/control_v.m`
速度模型约束函数，确保反演结果在物理合理范围内。

#### `external/extender21.m`
模型扩展函数，用于边界处理。

#### `external/ebcdic2ascii.m`和`external/altreadsegy.m`
SEG-Y格式数据读取和处理函数。

## 技术特点
1. **差分方案**
   - 使用交错网格有限差分方法
   - 二阶时间积分和四阶空间差分
   - 显式时间步进计算

2. **边界条件**
   - 标准PML吸收边界条件
   - 上边界自由表面条件处理
   - 模型扩展处理边界效应

3. **优化策略**
   - 共轭梯度法优化
   - 黄金分割线搜索
   - 震源子波同步估计

4. **目标函数选择**
   - 支持多种目标函数(L2波形残差、互相关残差等)
   - 可配置不同的预处理策略

## 模型参数
声波各向同性模型基于以下参数：
- `vp`: 纵波速度，决定介质中声波传播速度
- `rho`: 密度(在此实现中通常假定为常数)

## 波场计算详解
1. **声波方程系统**
   - 基于一阶声波方程(压力-速度耦合系统)
   - 中心差分格式离散化
   - 压力场和速度场交错更新

2. **PML实现**
   - 标准PML吸收边界实现
   - 速度场和压力场在PML区域特殊处理
   - 参数平滑变化确保吸收效果

3. **震源实现**
   - 默认使用Ricker子波
   - 支持自定义震源子波输入
   - 震源添加采用点源近似

## 函数式设计特点
1. **独立函数模块**
   - 每个功能点实现为独立函数
   - 通过参数传递实现数据交换
   - 无需维护复杂的类状态

2. **清晰的输入输出接口**
   - 函数参数列表明确定义所需输入
   - 返回值清晰表达计算结果
   - 函数间通过显式参数传递通信

3. **易于修改和扩展**
   - 可单独修改或替换特定功能模块
   - 直接添加新函数实现新功能
   - 便于理解和调试单个组件

### 波场数据结构对比

#### 声波FWI中的波场数据
1. **完整波场(pf/realp)**:
   - 维度：[nz, nx, Nt]
   - 内容：声波压力场的时空演化
   - 用途：用于伴随波场反传和梯度计算
   - 特点：单一物理量（压力场）

2. **检波器记录(seisv0/realrec_p)**:
   - 维度：[Nt, nx]
   - 内容：检波器位置的压力场时间序列
   - 用途：计算观测数据与合成数据的残差
   - 特点：仅记录检波器位置的数据

#### VTI-Elastic需要的对应结构
1. **完整波场需要扩展为**:
   - 维度：[nz, nx, Nt, ncomponents]
   - 内容：应力场（σxx, σzz, σxz）和速度场（vx, vz）
   - 用途：相同（伴随反传和梯度计算）
   - 特点：多个物理量的耦合演化

2. **检波器记录需要扩展为**:
   - 维度：[Nt, nx, ncomponents]
   - 内容：检波器位置的速度场分量（vx, vz）
   - 用途：相同（残差计算）
   - 特点：需要考虑多分量记录

### 目标函数和梯度计算对比

#### 目标函数计算
1. **声波FWI (obj1)**:
   - 使用归一化互相关目标函数
   - 单一物理量（压力场）的匹配
   - 计算公式：obj = -sum(sum(归一化互相关))

2. **VTI-Elastic (misfit)**:
   - 使用L2范数目标函数
   - 同时考虑vx和vy两个分量
   - 计算公式：misfit = 0.5 * (||residuals_vx||² + ||residuals_vy||²)

#### 梯度计算
1. **声波FWI (pb)**:
   - 直接在时间域累加波场互相关
   - 单一梯度分量（速度）
   - 计算公式：bp = sum(正演波场 * 伴随波场)

2. **VTI-Elastic (gradient)**:
   - 使用MEX函数优化的梯度计算
   - 多个弹性参数梯度：
     * c11梯度：-∂vx/∂x * ∂v†x/∂x
     * c13梯度：-(∂v†x/∂x * ∂vy/∂y + ∂v†y/∂y * ∂vx/∂x)
     * c33梯度：-∂vy/∂y * ∂v†y/∂y
     * c44梯度：-(∂vx/∂y + ∂vy/∂x) * (∂v†x/∂y + ∂v†y/∂x)
     * ρ梯度：-v†i * ∂²vi/∂t²

### 梯度计算的物理基础对比

1. **声波FWI中的梯度(img1)**:
   - 计算公式：gradient = (2/v^3) * correlation(forward, adjoint)
   - 系数来源：声波方程中速度参数v的二次方倒数形式
   - 特点：需要考虑速度的三次方作为系数

2. **VTI-Elastic中的梯度**:
   - 不需要参数的三次方项
   - 弹性参数直接出现在方程中
   - 梯度计算基于速度场的空间和时间导数
   - 使用MEX函数优化计算性能

### 目标函数值对比

1. **声波FWI (objval0)**:
   - 直接累加所有炮的目标函数值
   - 使用归一化互相关目标函数
   - 变量命名：objval0

2. **VTI-Elastic (current_misfit)**:
   - 使用L2范数目标函数
   - 同时考虑vx和vy两个分量
   - 变量命名：current_misfit
   - 计算公式：0.5 * (||residuals_vx||² + ||residuals_vy||²)

### 移植注意事项
1. **梯度计算**:
   - 不要移植声波的v^3系数
   - 保持VTI-Elastic原有的梯度计算公式
   - 使用MEX函数保证计算效率

2. **目标函数**:
   - 继续使用VTI-Elastic的L2范数目标函数
   - 保持对多分量残差的处理
   - 可以保留current_misfit的命名约定

## 与VTI-Elastic对比
1. **编程范式差异**
   - 声波项目: 函数式编程，独立函数模块
   - VTI-Elastic: 面向对象编程，类层次结构

2. **物理模型差异**
   - 声波项目: 各向同性声波介质(单参数vp)
   - VTI-Elastic: 各向异性弹性介质(多参数c11,c13,c33,c44,rho)

3. **功能组织差异**
   - 声波项目: 扁平结构，功能通过函数名区分
   - VTI-Elastic: 层次化结构，通过类和方法组织功能

4. **调用方式差异**
   - 声波项目: 直接函数调用，显式参数传递
   - VTI-Elastic: 创建对象实例，调用方法

## 代码移植指南

### 基本迁移策略
1. **功能映射**
   - 将每个独立函数映射到对应的类方法
   - 保持算法核心逻辑不变，调整接口匹配

2. **参数转换**
   - 函数参数转换为类属性或方法参数
   - 注意处理二者之间的参数名称和约定差异

3. **边界条件适配**
   - 将声波PML边界条件逻辑调整为VTI-Elastic中的CPML实现
   - 保持模型边界处理逻辑一致

### 具体迁移步骤

#### fd_p2D_fs.m → VTI_WaveFieldSolver/VTI_SingleShotModeling
1. 参数映射:
   - vp → c11,c33等弹性参数
   - npml → NPOINTS_PML
   - srcx,srcz → ISOURCE,JSOURCE
   - recx,recz → rec_x,rec_y

2. 边界处理:
   - 声波PML → VTI的CPML实现
   - 自由表面条件向VTI版本转换

3. 波场更新:
   - 声波压力-速度系统 → VTI的速度-应力系统
   - 考虑各向异性带来的额外耦合项

#### fd_p2D_fs_re.m → VTI_Adjoint类
1. 反传逻辑:
   - 时间反向遍历 → VTI_Adjoint中的时间反传
   - 残差作为伴随源 → VTI_Adjoint的add_adjoint_source方法

2. 伴随波场计算:
   - 将独立的伴随计算函数转换为类方法
   - 保持核心算法逻辑不变

#### fwi_CG_simple.m → VTI_FWI和优化器类
1. 优化算法:
   - 将共轭梯度法逻辑迁移到对应优化器类
   - 保持线搜索和收敛检查逻辑

2. 迭代控制:
   - 函数内循环 → VTI_FWI的run方法
   - 参数更新逻辑调整

#### 目标函数迁移
1. 将objfun1,objfun2,cross_resid等移植为目标函数类的方法
2. 保持计算逻辑不变，调整接口匹配VTI-Elastic需求

### 注意事项
1. **物理参数转换**
   - 声波速度vp需映射到多个VTI弹性参数
   - 需要额外定义剪切波速度相关参数
   - 考虑各向异性比例关系

2. **边界条件差异**
   - 声波PML和VTI的CPML尽管概念类似但实现有差异
   - 各向异性带来的额外稳定性要求

3. **计算效率考虑**
   - 弹性VTI计算量大于声波模拟
   - 可能需要额外优化以保持性能

4. **震源表示差异**
   - 声波点源转换为VTI中的力矢量震源
   - 考虑力的角度(ANGLE_FORCE)设置

5. **接口一致性**
   - 确保迁移后的方法维持与类结构一致的接口
   - 避免引入过多依赖，保持模块化

## 结论
声波各向同性FWI项目采用函数式编程范式实现了完整的声波波场模拟和反演功能。在移植到VTI-Elastic项目时，需要将其函数式结构适配到面向对象框架中，同时考虑从声波到弹性波、从各向同性到各向异性的物理模型扩展。保持算法核心逻辑不变的同时，调整接口和数据流以匹配VTI-Elastic的类结构，将有助于顺利完成代码迁移。

### 目标函数计算对比补充

1. **声波FWI中的总目标函数计算**:
   ```matlab
   objval0 = sum(obj0);  % 直接累加所有炮的目标函数值
   ```

2. **VTI-Elastic中的总目标函数计算**:
   ```matlab
   % 在VTI_FWI.compute_misfit_and_gradient中
   total_misfit = 0;
   for ishot = 1:obj.nshots
       [shot_gradient, shot_misfit] = obj.gradient_calculator.compute_single_shot_gradient(ishot, forward_wavefield);
       total_misfit = total_misfit + shot_misfit;
   end
   ```

3. **主要区别**:
   - 声波代码：在一个函数中直接累加
   - VTI-Elastic：分层实现
     * VTI_Adjoint计算单炮目标函数值
     * VTI_FWI累加所有炮的目标函数值
   - 两种实现在数学上等价，但VTI-Elastic的实现更模块化 

### 梯度处理对比补充

1. **声波FWI中的梯度处理**:
   ```matlab
   % 简单的水层梯度置零
   grad_stk1(1:water,:) = 0;  % grad_stk1表示叠加后的总梯度
   ```

2. **VTI-Elastic中的梯度处理**:
   ```matlab
   % 更灵活的水层处理机制
   function gradient = apply_water_layer_mask(obj, gradient)
       water_layer = obj.model_constraints.water_layer;
       if water_layer > 0
           fields = fieldnames(gradient);
           for i = 1:length(fields)
               field = fields{i};
               gradient.(field)(1:water_layer, :) = 0;
           end
       end
   end
   ```

3. **主要改进**:
   - 面向对象的实现更灵活
   - 可通过参数控制是否启用水层处理
   - 自动处理所有弹性参数的水层梯度
   - 保持了代码的扩展性 