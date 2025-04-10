# VTI-Elastic 项目提示文档

## 项目概述
VTI-Elastic 是一个用于垂直横向各向同性(VTI)介质中弹性波传播模拟与全波形反演(FWI)的MATLAB框架。该项目实现了完整的正演模拟、伴随状态方法梯度计算和多种优化算法，用于地震数据处理和地下结构成像。

## 核心功能
1. **波场模拟**
   - 实现了VTI介质中的弹性波传播模拟
   - 支持PML吸收边界条件
   - 包含速度场(vx, vy)和应力场(sigmaxx, sigmayy, sigmaxy)的计算
   - 提供多种计算模式：纯CPU、CPU-MEX加速、SIMD加速

2. **全波形反演(FWI)**
   - 实现了完整的FWI工作流
   - 基于伴随状态方法计算梯度
   - 支持多种目标函数(波形残差L2范数、互相关残差)
   - 提供多种优化算法(梯度下降法、L-BFGS法、Fletcher-Reeves共轭梯度法)

## 项目结构及核心组件详解

### `core/VTI_WaveFieldSolver.m`
波场求解器的核心实现，专注于VTI介质中波场传播的物理计算。

#### 类定义和结构
```matlab
classdef VTI_WaveFieldSolver < handle
    properties
        % 网格、材料、震源、检波器参数及波场变量
    end
    
    methods
        % 波场计算与物理模拟相关方法
    end
end
```

#### 属性详解
- **网格参数**:
  - `NX`, `NY`：x和y方向网格点数，定义计算域大小
  - `DELTAX`, `DELTAY`：x和y方向网格间距，单位通常为米
  
- **材料参数**:
  - `c11`, `c13`, `c33`, `c44`：VTI介质的弹性常数，单位为帕斯卡(Pa)
  - `rho`：密度，单位为千克/立方米(kg/m³)
  - `f0`：震源频率，单位为赫兹(Hz)
  
- **PML参数**:
  - `PML_XMIN`, `PML_XMAX`, `PML_YMIN`, `PML_YMAX`：控制四个边界是否启用PML
  - `NPOINTS_PML`：PML层的厚度（网格点数）
  - `NPOWER`, `K_MAX_PML`, `ALPHA_MAX_PML`：控制PML吸收效果的参数
  - 各种PML记忆变量和系数：如`d_x`, `K_x`, `alpha_x`等
  
- **震源参数**:
  - `ISOURCE`, `JSOURCE`：震源在网格上的位置索引（整数值）
  - `xsource`, `ysource`：震源的物理坐标（浮点值）
  - `t0`：时间延迟，控制震源激发时间
  - `factor`：震源强度因子
  - `ANGLE_FORCE`：力的角度，单位为度
  
- **检波器参数**:
  - `NREC`：检波器数量
  - `first_rec_x`, `first_rec_y`：第一个检波器的位置
  - `rec_x`, `rec_y`：所有检波器的位置数组
  - `rec_dx`, `rec_dy`：检波器间距
  
- **波场变量**:
  - `vx`, `vy`：速度场分量
  - `sigmaxx`, `sigmayy`, `sigmaxy`：应力场分量
  - 各种记忆变量用于PML实现

#### 方法详解
- **构造与初始化**
  - `VTI_WaveFieldSolver(params)`：构造函数，接收参数结构体并初始化
  - `initialize(params)`：初始化所有参数和数组

- **边界条件**
  - `setup_pml_boundary()`：验证VTI材料在PML区域的稳定性条件
  - `setup_pml_boundary_x()`：设置x方向的PML参数
  - `setup_pml_boundary_y()`：设置y方向的PML参数
  - `apply_boundary_conditions()`：应用刚性边界条件

- **波场传播**
  - `compute_wave_propagation()`：根据计算模式调用不同的计算函数
  - `add_source(it)`：在当前时间步添加震源
  - `record_seismograms(it)`：记录当前时间步检波器位置的波场值

- **辅助功能**
  - `setup_receivers()`：初始化检波器位置数组
  - `reset_fields()`：重置所有波场和记忆变量
  - `output_info(it)`：输出计算状态信息和稳定性检查

#### 重要改动说明
- 移除了与文件存储和炮号管理相关的功能
- 移除了`save_snapshots`、`output_dir`和`current_shot_number`属性
- 移除了`set_current_shot`方法
- 修改了`output_info`方法，仅保留计算信息输出
- 去除了多炮记录的数组存储

### `core/VTI_SingleShotModeling.m`
单炮正演模拟的高层封装，负责管理完整的单炮模拟流程、数据存储和震源位置管理。

#### 类定义和结构
```matlab
classdef VTI_SingleShotModeling < handle
    properties
        % 基本参数、震源位置参数、求解器实例、输出目录
    end
    
    methods
        % 高层模拟流程控制方法
    end
end
```

#### 属性详解
- **基本参数**:
  - `NSTEP`：时间步数
  - `NREC`：检波器数量
  - `DELTAT`：时间步长
  
- **震源位置参数**:
  - `first_shot_x`, `first_shot_y`：第一个炮点的网格索引（坐标系命名更直观）
  - `shot_dx`, `shot_dy`：炮点间隔（网格点数）
  
- **求解器实例**:
  - `fd_solver`：VTI_WaveFieldSolver类的实例
  
- **输出和状态**:
  - `output_dir`：输出基础目录
  - `seismogram_dir`：地震记录保存目录
  - `current_shot_number`：当前炮号（从WaveFieldSolver移至此处）

#### 方法详解
- **构造与初始化**
  - `VTI_SingleShotModeling(params)`：构造函数，创建并初始化波场求解器
  
- **正演模拟**
  - `forward_modeling_single_shot(ishot)`：执行单炮正演模拟，完整流程
    ```matlab
    % 主要步骤：
    % 1. 更新当前炮号
    % 2. 设置PML边界
    % 3. 设置震源位置
    % 4. 时间步进计算
    % 5. 保存地震记录和完整波场
    ```
  - `set_source_position(ishot)`：根据炮号设置震源位置
    ```matlab
    % 计算炮点位置并更新波场求解器的震源参数
    obj.fd_solver.ISOURCE = obj.first_shot_x + (ishot-1) * obj.shot_dx;
    obj.fd_solver.JSOURCE = obj.first_shot_y + (ishot-1) * obj.shot_dy;
    ```
  - `update_model_params(model)`：更新模型参数（用于FWI迭代）

#### 重要改动说明
- 震源位置参数名称从`i/j`改为`x/y`，增加直观性
- 移除了波场快照保存相关功能
- 移除了多炮记录数组存储
- 炮号管理功能从WaveFieldSolver移至此类
- 简化了文件存储，仅保留关键的地震记录保存
- 精简了不必要的方法，专注于FWI所需的关键功能

### `core/VTI_Adjoint.m`
伴随波场计算实现，用于全波形反演中的梯度计算。

#### 类定义和结构
```matlab
classdef VTI_Adjoint < handle
    properties
        % 波场求解器、模型参数、残差和波场存储
    end
    
    methods
        % 各种方法实现
    end
end
```

#### 属性详解
- **波场求解器**:
  - `wavefield_solver_obs`：观测数据的波场求解器
  - `wavefield_solver_syn`：合成数据的波场求解器
  
- **模型参数**:
  - `obs_params`：观测数据的模型参数
  - `syn_params`：合成数据的模型参数
  
- **残差和数据存储**:
  - `current_residuals_vx`、`current_residuals_vy`：当前炮的速度场残差
  - `current_forward_wavefield`：当前炮的正演波场

#### 方法详解
- **构造与初始化**
  - `VTI_Adjoint(obs_params, syn_params)`：构造函数，创建两个波场求解器
    ```matlab
    % 创建观测数据和合成数据的波场求解器
    obj.wavefield_solver_obs = VTI_SingleShotModeling(obs_params);
    obj.wavefield_solver_syn = VTI_SingleShotModeling(syn_params);
    ```

- **伴随波场计算**
  - `compute_residuals_single_shot(ishot)`：计算单炮残差
    ```matlab
    % 计算观测数据和合成数据的差值
    obj.current_residuals_vx = vx_obs - vx_syn;
    obj.current_residuals_vy = vy_obs - vy_syn;
    ```
  - `compute_adjoint_wavefield_single_shot(ishot)`：计算单炮伴随波场
    ```matlab
    % 主要步骤：
    % 1. 计算残差
    % 2. 时间反传计算伴随波场
    ```
  - `compute_adjoint_source(it)`：计算伴随源项

- **目标函数计算**
  - `compute_misfit_function(ishot)`：计算目标函数值
    ```matlab
    % 计算残差的L2范数
    misfit = 0.5 * sum(sum(residuals_vx.^2 + residuals_vy.^2));
    ```

#### 调用示例
```matlab
% 创建伴随波场计算器
adjoint_solver = VTI_Adjoint(obs_params, syn_params);

% 计算伴随波场
adjoint_wavefield = adjoint_solver.compute_adjoint_wavefield_single_shot(1);
```

#### 移植注意事项
- 确保残差计算的正确性
- 注意伴随源的添加方式
- 时间反传计算的稳定性

### `core/VTI_Gradient.m`
基于伴随状态方法的梯度计算模块，计算模型参数的梯度。

#### 类定义和结构
```matlab
classdef VTI_Gradient < handle
    properties
        % 伴随求解器和梯度参数
    end
    
    methods
        % 各种方法实现
    end
end
```

#### 属性详解
- **伴随求解器**:
  - `adjoint_solver`：VTI_Adjoint类的实例

#### 方法详解
- **构造与初始化**
  - `VTI_Gradient(adjoint_solver)`：构造函数，接收伴随求解器
    ```matlab
    obj.adjoint_solver = adjoint_solver;
    ```

- **梯度计算**
  - `compute_gradient_single_shot(ishot)`：计算单炮梯度
    ```matlab
    % 主要步骤：
    % 1. 计算正演波场
    % 2. 计算伴随波场
    % 3. 波场互相关计算梯度
    ```
  - `compute_gradient_all_shots()`：计算所有炮的梯度和
    ```matlab
    % 循环计算每炮梯度并累加
    for ishot = 1:adjoint_solver.syn_params.NSHOT
        gradient_shot = obj.compute_gradient_single_shot(ishot);
        total_gradient = total_gradient + gradient_shot;
    end
    ```
  - `correlate_wavefields(forward_wavefield, adjoint_wavefield)`：波场互相关计算梯度
    ```matlab
    % 时空域上的波场互相关
    for it = 1:NT
        % 各参数梯度的更新计算
    end
    ```

- **辅助功能**
  - `compute_gradient(field)`：计算场量的空间导数
    ```matlab
    % 使用有限差分计算梯度
    dx = zeros(size(field));
    dy = zeros(size(field));
    % 计算导数...
    ```
  - `velocity_to_displacement(vx, vy, dt)`：速度场转位移场
    ```matlab
    % 时间积分转换
    ux = cumsum(vx, 3) * dt;
    uy = cumsum(vy, 3) * dt;
    ```

#### 调用示例
```matlab
% 创建梯度计算器
gradient_calculator = VTI_Gradient(adjoint_solver);

% 计算单炮梯度
gradient = gradient_calculator.compute_gradient_single_shot(1);

% 计算所有炮的梯度
total_gradient = gradient_calculator.compute_gradient_all_shots();
```

#### 移植注意事项
- 注意梯度计算中的波场互相关方法
- 确保空间导数计算的精度
- 注意不同参数梯度的物理单位

### `core/VTI_FWI.m`
全波形反演的主控制模块，集成优化算法进行模型参数反演。

#### 类定义和结构
```matlab
classdef VTI_FWI < handle
    properties
        % 优化器
    end
    
    methods
        % 各种方法实现
    end
end
```

#### 属性详解
- **优化器**:
  - `optimizer`：BaseOptimizer的子类实例

#### 方法详解
- **构造与初始化**
  - `VTI_FWI(params)`：构造函数，根据参数选择优化器
    ```matlab
    % 根据params.optimization选择不同优化器
    switch params.optimization
        case 'gradient_descent'
            obj.optimizer = GradientDescentOptimizer(params);
        case 'LBFGS'
            obj.optimizer = LBFGSOptimizer(params);
        % 其他类型...
    end
    ```

- **优化执行**
  - `run()`：执行FWI优化过程
    ```matlab
    % 调用优化器的run方法
    obj.optimizer.run();
    ```

#### 调用示例
```matlab
% 创建FWI控制器
params.optimization = 'LBFGS';  % 选择优化算法
fwi_controller = VTI_FWI(params);

% 执行FWI
fwi_controller.run();
```

#### 移植注意事项
- 确保与不同优化器的接口一致性
- 注意参数结构体的完整性

## 技术特点
1. **差分方案**
   - 采用二阶交错网格有限差分方法
   - 高效实现VTI介质中的波动方程

2. **边界条件**
   - 使用CPML(Convolutional Perfectly Matched Layer)完美匹配层
   - 有效抑制人工边界反射
   - 实现了Becache等(2003)提出的PML稳定性条件验证

3. **计算优化**
   - 提供多种计算模式(CPU, MEX, SIMD)
   - 针对大规模计算进行了内存优化
   - 支持OpenMP并行和SIMD指令集优化

4. **VTI参数体系**
   - 基于完整的弹性参数集(c11, c13, c33, c44, rho)
   - 支持复杂VTI介质模型
   - 包含参数合法性验证

## 模型参数
VTI介质模型基于以下参数：
- `c11`, `c13`, `c33`, `c44`: VTI介质的弹性常数
- `rho`: 密度
- `epsilon`, `delta`: Thomsen参数(在工具中用于模型生成)

VTI模型的物理含义：
- c11: 横向P波模量，控制水平传播的P波速度
- c33: 纵向P波模量，控制垂直传播的P波速度
- c13: 耦合模量，影响倾斜角度的P波速度
- c44: 剪切模量，控制S波速度
- rho: 介质密度

## 波场计算详解
1. **速度-应力耦合系统**
   - 速度场(vx, vy)和应力场(sigmaxx, sigmayy, sigmaxy)通过交错时间步更新
   - 利用CPML吸收边界处理波场与边界的相互作用

2. **PML实现细节**
   - 基于CPML(Convolutional Perfectly Matched Layer)实现
   - 通过记忆变量(memory_dvx_dx, memory_dvx_dy等)存储吸收边界的历史作用
   - 边界参数(d_x, K_x, alpha_x等)按照距离边界的位置平滑变化

3. **震源实现**
   - 采用Ricker子波作为震源
   - 支持任意角度的力矢量震源
   - 震源强度和时间延迟可调控

## 代码修改注意事项
1. **保持数组维度一致**
   - 模型参数和波场变量的维度必须匹配
   - 注意MATLAB的列优先存储与代码中的索引关系
   - VTI_WaveFieldSolver对数组大小有严格检查

2. **内存管理**
   - 大规模波场计算时注意内存使用
   - 对于完整波场存储，可考虑磁盘存储模式
   - 波场快照保存受save_snapshots参数控制

3. **边界条件维护**
   - 修改波场更新代码时需同步考虑PML边界条件
   - PML参数调整需谨慎，影响数值稳定性
   - 需保持Becache稳定性条件验证流程

4. **计算模式兼容**
   - 修改核心计算代码时需考虑所有计算模式的兼容性
   - MEX函数修改需重新编译相应模块
   - 注意CPU和SIMD版本的算法保持一致

5. **优化算法参数**
   - 不同优化算法有特定参数设置，修改时需参考相应文档
   - 步长和收敛判断标准对优化过程至关重要

## 代码规范与移植指南

### 命名规范
1. **类名**：使用大驼峰命名法，如`VTI_WaveFieldSolver`
2. **方法名**：使用小驼峰命名法，如`computeWavePropagation`
3. **私有变量**：使用下划线前缀，如`_privateVar`

### 变量规范
1. **常量**：全部大写，单词间用下划线分隔，如`MAX_ITERATIONS`
2. **一般变量**：使用小驼峰命名法，如`velocityField`
3. **布尔变量**：使用is/has前缀，如`isStable`, `hasConverged`

### 文档规范
1. **类文档**：在类定义前提供完整的类功能说明
2. **方法文档**：对每个公共方法提供输入、输出和功能说明
3. **参数文档**：对重要参数的物理意义和单位进行说明

### 移植步骤指南
1. **环境准备**：确保目标环境具备必要的MATLAB版本和工具箱
2. **框架移植**：先移植核心类和结构，确保基本框架可运行
3. **功能迁移**：逐步迁移各功能模块，每步测试确保正确性
4. **优化适配**：根据目标环境优化性能，如调整计算模式和内存使用
5. **集成测试**：进行端到端测试，验证整个流程的正确性

### 跨平台注意事项
1. **路径处理**：使用平台无关的路径处理方式，如`fullfile`函数
2. **内存管理**：大规模数据考虑使用磁盘存储或分批处理
3. **并行计算**：注意不同平台的并行计算库差异

### 扩展开发指南
1. **新增参数**：保持与现有参数结构一致
2. **添加功能**：遵循现有类的设计模式
3. **性能优化**：先分析瓶颈再优化，避免过早优化

## 待优化功能
1. **实际数据处理**
   - 集成SEG-Y格式数据读取功能
   - 实现数据预处理和质量控制

2. **震源子波提取**
   - 从实际数据中反演提取震源子波
   - 实现自适应时窗处理

3. **目标函数优化**
   - 实现基于包络的伴随源计算
   - 添加包络归一化的互相关计算

4. **GPU加速**
   - 为计算密集型任务添加GPU加速支持

## 参考资料
- VTI介质理论基础可参考Thomsen(1986)的经典论文
- 伴随状态方法实现参考Tromp et al.(2005)和Plessix(2006)
- 数值模拟方法可参考Virieux(1986)和Robertsson et al.(1994)
- PML边界条件参考Komatitsch和Martin(2007)的CPML实现

## 共轭梯度法移植指南

### 核心组件对比
1. **目标函数计算**
   - 声波项目：单一参数(vp)的L2范数残差
   - VTI项目：多参数(c11,c13,c33,c44,rho)的联合残差

2. **梯度计算**
   - 声波项目：单一梯度场
   - VTI项目：五个参数的梯度场，需要考虑参数间的耦合关系

3. **步长控制**
   - 声波项目：单一步长控制
   ```matlab
   while maxd > 30
       a = dec*a;
       maxd = max(max(abs(a*d1)));
   end
   ```
   - VTI项目：需要为不同参数设置独立的步长控制
   ```matlab
   % 示例：不同参数的步长控制
   max_update = struct('c11', 30, 'c13', 20, 'c33', 30, 'c44', 15, 'rho', 100);
   ```

4. **搜索方向计算**
   - 声波项目：标量场的Fletcher-Reeves公式
   ```matlab
   b = sqrt(sum(sum((p'*p).^2))/sum(sum((p0'*p0).^2)));
   d1 = -p + b*d0;
   ```
   - VTI项目：多参数场的扩展形式
   ```matlab
   % 需要考虑多个参数的梯度组合
   b = compute_FR_beta(current_grad, previous_grad);
   d = compute_conjugate_direction(current_grad, b, previous_d);
   ```

### 实现策略
1. **类结构设计**
```matlab
classdef VTI_ConjugateGradient < handle
    properties
        % 优化参数
        max_iterations
        tolerance
        line_search_params
        
        % 步长控制
        step_length_controls
        
        % 梯度和方向存储
        previous_gradient
        previous_direction
        
        % 模型约束
        model_constraints
    end
    
    methods
        % 主优化循环
        function [optimal_model, history] = optimize(obj, initial_model)
        
        % Fletcher-Reeves系数计算
        function beta = compute_FR_coefficient(obj, current_grad, previous_grad)
        
        % 线搜索实现
        function [step_length, new_model] = line_search(obj, model, direction)
        
        % 模型更新和约束
        function new_model = update_model(obj, model, direction, step_length)
    end
end
```

2. **关键算法实现**
```matlab
% 主优化循环示例
function [optimal_model, history] = optimize(obj, initial_model)
    current_model = initial_model;
    k = 1;
    
    while k <= obj.max_iterations
        % 1. 计算目标函数和梯度
        [f, grad] = obj.compute_objective_and_gradient(current_model);
        
        % 2. 计算搜索方向
        if k == 1
            direction = -grad;  % 第一次使用最速下降
        else
            beta = obj.compute_FR_coefficient(grad, obj.previous_gradient);
            direction = -grad + beta * obj.previous_direction;
        end
        
        % 3. 线搜索
        [step_length, new_model] = obj.line_search(current_model, direction);
        
        % 4. 更新模型
        current_model = obj.update_model(current_model, direction, step_length);
        
        % 5. 存储历史
        obj.previous_gradient = grad;
        obj.previous_direction = direction;
        
        k = k + 1;
    end
end
```

### 注意事项
1. **梯度预处理**
   - 不同参数的梯度可能量级差异很大
   - 需要合适的归一化策略
   - 考虑使用预条件子提高收敛性

2. **线搜索策略**
   - 多参数情况下可能需要更复杂的线搜索
   - 考虑参数间的相互影响
   - 可能需要自适应步长调整

3. **收敛判断**
   - 需要综合考虑多个参数的收敛情况
   - 可以设置不同参数的收敛阈值
   - 考虑目标函数和梯度范数的变化

## 波场存储与内存管理

### 存储策略
1. **完整波场存储**
   - 用于梯度计算的正演波场
   - 时间维度完整存储
   - 考虑内存限制可能需要分批处理

2. **检查点策略**
   - 在关键时间步存储完整波场
   - 其他时间步重新计算
   - 平衡计算时间和内存使用

3. **数据压缩**
   - 考虑波场数据的压缩存储
   - 可以使用单精度存储非关键数据
   - 权衡精度和存储空间

## 性能优化指南

### 计算优化
1. **并行计算**
   - 多炮并行策略
   - 波场计算的OpenMP优化
   - 梯度计算的并行实现

2. **内存访问优化**
   - 合理的数组布局
   - 缓存友好的访问模式
   - 避免不必要的数据复制

3. **I/O优化**
   - 异步I/O操作
   - 数据批量读写
   - 合理的存储格式选择

## 调试与验证指南

### 验证步骤
1. **梯度验证**
   - 使用有限差分验证梯度计算
   - 检查梯度的对称性
   - 验证伴随波场的正确性

2. **优化验证**
   - 检查目标函数单调下降
   - 验证搜索方向的共轭性
   - 确认步长选择的合理性

3. **结果分析**
   - 模型更新的物理合理性
   - 收敛速度分析
   - 不同参数的敏感性分析 