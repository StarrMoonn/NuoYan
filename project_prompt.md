# VTI-Elastic 项目最新文档

## 项目概述
VTI-Elastic 是一个用于垂直横向各向同性(VTI)介质中弹性波传播模拟与全波形反演(FWI)的MATLAB框架。该项目实现了完整的正演模拟、伴随状态方法梯度计算和Fletcher-Reeves共轭梯度优化算法，用于地震数据处理和地下结构成像。

## 核心模块详解

### `core/VTI_WaveFieldSolver.m`
波场求解器的核心实现，负责VTI介质中物理波场的数值计算。

#### 主要属性
- **网格参数**: `NX`, `NY`, `DELTAX`, `DELTAY`
- **材料参数**: `c11`, `c13`, `c33`, `c44`, `rho`
- **边界条件**: `PML_XMIN`, `PML_XMAX`, `PML_YMIN`, `PML_YMAX`, `NPOINTS_PML`
- **波场变量**: `vx`, `vy`, `sigmaxx`, `sigmayy`, `sigmaxy`

#### 核心方法
- **构造函数**: `obj = VTI_WaveFieldSolver(params)`
  - 输入: 参数结构体
  - 输出: 求解器实例

- **波场计算**: `compute_wave_propagation()`
  - 功能: 计算一个时间步的波场传播
  - 输出: 无直接返回值，更新内部波场变量

- **震源添加**: `add_source(it)`
  - 输入: 时间步索引
  - 功能: 在当前时间步添加震源项

- **边界设置**:
  - `setup_pml_boundary()`: 设置并验证PML参数
  - `setup_pml_boundary_x()`: 设置x方向PML
  - `setup_pml_boundary_y()`: 设置y方向PML
  - `apply_boundary_conditions()`: 应用边界条件

- **重要辅助方法**:
  - `reset_fields()`: 重置所有波场变量
  - `output_info(it)`: 输出计算信息

### `core/VTI_SingleShotModeling.m`
单炮正演模拟的高层封装，管理完整的单炮模拟流程。

#### 主要属性
- **基本参数**: `NSTEP`, `NREC`, `DELTAT`, `IT_DISPLAY`
- **震源参数**: `first_shot_x`, `first_shot_y`, `shot_dx`, `shot_dy`
- **求解器实例**: `fd_solver`
- **输出设置**: `output_dir`, `seismogram_dir`, `current_shot_number`

#### 核心方法
- **构造函数**: `obj = VTI_SingleShotModeling(params)`
  - 输入: 参数结构体
  - 输出: 单炮模拟实例

- **正演模拟**: `[vx_data, vy_data, complete_wavefield] = forward_modeling_single_shot(ishot)`
  - 输入: 炮号
  - 输出:
    - `vx_data`: 水平分量地震记录 [NSTEP x NREC]
    - `vy_data`: 垂直分量地震记录 [NSTEP x NREC]
    - `complete_wavefield`: 完整波场结构体 {vx, vy} [NX x NY x NSTEP]

- **震源设置**: `set_source_position(ishot)`
  - 输入: 炮号
  - 功能: 根据炮号设置震源位置

- **模型更新**: `update_model_params(model)`
  - 输入: 模型参数结构体
  - 功能: 更新求解器的模型参数（用于FWI迭代）

### `core/VTI_Adjoint.m`
伴随波场计算实现，用于计算梯度所需的伴随波场。

#### 主要属性
- **波场求解器**: `wavefield_solver_obs`, `wavefield_solver_syn`
- **参数结构体**: `obs_params`, `syn_params`
- **残差数据**: `current_residuals_vx`, `current_residuals_vy`
- **时间参数**: `NSTEP`, `DELTAT`

#### 核心方法
- **构造函数**: `obj = VTI_Adjoint(params)`
  - 输入: 包含obs_params和syn_params的参数结构体
  - 输出: 伴随计算实例

- **残差计算**: `[misfit] = compute_residuals_single_shot(ishot)`
  - 输入: 炮号
  - 输出: 
    - `misfit`: 目标函数值（L2范数）
    - 副作用: 更新内部残差变量

- **伴随波场计算**: `[adjoint_wavefield] = compute_adjoint_wavefield_single_shot(ishot)`
  - 输入: 炮号
  - 输出: 
    - `adjoint_wavefield`: 伴随波场结构体 {vx, vy} [NX x NY x NSTEP]

- **伴随源添加**: `add_adjoint_source(wave_solver, residuals_vx, residuals_vy, it)`
  - 输入: 波场求解器实例，残差数据，时间步
  - 功能: 向波场添加伴随源

### `core/VTI_Gradient.m`
梯度计算模块，基于正演波场和伴随波场计算模型参数梯度。

#### 主要属性
- **伴随求解器**: `adjoint_solver`
- **参数设置**: `gradient_output_dir`, `NSTEP`, `save_shot_gradient`

#### 核心方法
- **构造函数**: `obj = VTI_Gradient(params)`
  - 输入: 参数结构体
  - 输出: 梯度计算实例

- **单炮梯度计算**: `[gradient, misfit] = compute_single_shot_gradient(ishot, forward_wavefield)`
  - 输入: 炮号，正演波场
  - 输出: 
    - `gradient`: 梯度结构体 {c11, c13, c33, c44, rho}
    - `misfit`: 目标函数值

- **VTI梯度计算**: `gradient = compute_vti_gradient(forward_wavefield, adjoint_wavefield)`
  - 输入: 正演波场，伴随波场
  - 输出: 梯度结构体 {c11, c13, c33, c44, rho}

- **梯度保存**: `save_gradient(gradient, ishot)`
  - 输入: 梯度结构体，炮号
  - 功能: 保存梯度到文件

### `core/VTI_FWI.m`
全波形反演主控制器，实现Fletcher-Reeves共轭梯度法优化。

#### 主要属性
- **基本依赖**: `modeling`, `adjoint`, `gradient_calculator`
- **优化参数**: `max_iterations`, `step_length`, `step_length_decay`, `max_line_search`
- **迭代状态**: `current_model`, `current_misfit`, `current_gradient`, `search_direction`
- **共轭梯度相关**: `previous_gradient`, `previous_direction`, `beta`
- **输出控制**: `output_dir`, `model_output_dir`, `gradient_output_dir`, `save_interval`

#### 核心方法
- **构造函数**: `obj = VTI_FWI(params)`
  - 输入: 参数结构体
  - 输出: FWI控制器实例

- **目标函数与梯度计算**: `[misfit, gradient] = compute_misfit_and_gradient(model)`
  - 输入: 模型参数结构体
  - 输出: 
    - `misfit`: 总目标函数值
    - `gradient`: 总梯度结构体

- **Fletcher-Reeves共轭梯度优化**:
  - `run_optimization()`: 执行完整优化流程
  - `perform_line_search()`: 执行线搜索
  - `beta = compute_FR_coefficient(current_grad, previous_grad)`: 计算Fletcher-Reeves系数
  - `direction = calculate_new_search_direction(grad, beta, prev_dir)`: 计算新的搜索方向

- **模型更新与约束**:
  - `new_model = update_model(model, direction, alpha)`: 更新模型参数
  - `model = apply_model_constraints(model)`: 应用模型物理约束
  - `gradient = apply_water_layer_mask(gradient)`: 应用水层掩码

- **可视化与输出**:
  - `save_iteration_history()`: 保存迭代历史
  - `plot_convergence()`: 绘制收敛曲线

## Fletcher-Reeves共轭梯度法实现详解

### 主要步骤
1. **目标函数与梯度计算**
   ```matlab
   [obj.current_misfit, obj.current_gradient] = obj.compute_misfit_and_gradient(obj.current_model);
   ```

2. **第一次迭代（最速下降方向）**
   ```matlab
   p = obj.normalize_gradient(obj.current_gradient);
   obj.search_direction = obj.negate_gradient(p);
   ```

3. **线搜索**
   ```matlab
   a = obj.step_length;
   while ks <= obj.max_line_search
       trial_model = obj.update_model(obj.current_model, obj.search_direction, a);
       [trial_misfit, trial_gradient] = obj.compute_misfit_and_gradient(trial_model);
       
       if trial_misfit < obj.current_misfit
           % 接受更新
           break;
       else
           a = obj.step_length_decay * a;
           ks = ks + 1;
       end
   end
   ```

4. **计算Fletcher-Reeves系数**
   ```matlab
   beta = obj.compute_FR_coefficient(obj.current_gradient, obj.previous_gradient);
   ```

5. **计算新的搜索方向**
   ```matlab
   direction = obj.calculate_new_search_direction(p, beta, obj.previous_direction);
   ```

### 多参数处理
VTI介质有五个参数(c11, c13, c33, c44, rho)，每个参数单独处理：

1. **梯度归一化**
   ```matlab
   normalized = struct();
   fields = fieldnames(gradient);
   for i = 1:length(fields)
       field = fields{i};
       normalized.(field) = 100 * gradient.(field) / norm_val;
   end
   ```

2. **梯度范数计算**
   ```matlab
   norm_sq = 0;
   fields = fieldnames(gradient);
   for i = 1:length(fields)
       field = fields{i};
       norm_sq = norm_sq + sum(gradient.(field)(:).^2);
   end
   norm_val = sqrt(norm_sq);
   ```

3. **Fletcher-Reeves系数计算**
   ```matlab
   current_norm = 0;
   previous_norm = 0;
   for i = 1:length(fields)
       field = fields{i};
       current_norm = current_norm + sum(sum(current_grad.(field).^2));
       previous_norm = previous_norm + sum(sum(previous_grad.(field).^2));
   end
   beta = current_norm / previous_norm;
   ```

4. **步长控制**
   ```matlab
   max_allowed = struct('c11', 30e9, 'c13', 20e9, 'c33', 30e9, 'c44', 15e9, 'rho', 100);
   for i = 1:length(fields)
       field = fields{i};
       max_updates.(field) = max(abs(direction.(field)(:))) * alpha;
       while max_updates.(field) > max_allowed.(field)
           alpha = alpha * dec_factor;
           max_updates.(field) = max(abs(direction.(field)(:))) * alpha;
       end
   end
   ```

### 物理约束处理
1. **参数范围约束**
   ```matlab
   model.c11 = max(min(model.c11, obj.model_constraints.c11_max), obj.model_constraints.c11_min);
   model.c13 = max(min(model.c13, obj.model_constraints.c13_max), obj.model_constraints.c13_min);
   model.c33 = max(min(model.c33, obj.model_constraints.c33_max), obj.model_constraints.c33_min);
   model.c44 = max(min(model.c44, obj.model_constraints.c44_max), obj.model_constraints.c44_min);
   model.rho = max(min(model.rho, obj.model_constraints.rho_max), obj.model_constraints.rho_min);
   ```

2. **稳定性约束**
   ```matlab
   stability_mask = (model.c11.*model.c33 - model.c13.^2) <= 0;
   if any(stability_mask(:))
       model.c13(stability_mask) = sqrt(0.99*model.c11(stability_mask).*model.c33(stability_mask));
   end
   ```

3. **水层掩码**
   ```matlab
   if water_layer > 0
       fields = fieldnames(gradient);
       for i = 1:length(fields)
           field = fields{i};
           gradient.(field)(1:water_layer, :) = 0;
       end
   end
   ```

## 梯度计算详解

### 单炮梯度计算
```matlab
function [gradient, misfit] = compute_single_shot_gradient(obj, ishot, forward_wavefield)
    % 先计算残差和目标函数值
    misfit = obj.adjoint_solver.compute_residuals_single_shot(ishot);
    
    % 再计算伴随波场
    adjoint_wavefield = obj.adjoint_solver.compute_adjoint_wavefield_single_shot(ishot);
    
    % 使用正演波场和伴随波场计算梯度
    gradient = obj.compute_vti_gradient(forward_wavefield, adjoint_wavefield);
end
```

### 多炮梯度累加
```matlab
% 初始化总梯度和目标函数值
total_gradient = struct('c11', zeros(size(model.c11)), ...
                      'c13', zeros(size(model.c13)), ...
                      'c33', zeros(size(model.c33)), ...
                      'c44', zeros(size(model.c44)), ...
                      'rho', zeros(size(model.rho)));
total_misfit = 0;

% 累加所有炮的梯度和目标函数值
for ishot = 1:obj.nshots
    % 正演模拟
    [~, ~, forward_wavefield] = obj.modeling.forward_modeling_single_shot(ishot);
    
    % 计算梯度和目标函数值
    [shot_gradient, shot_misfit] = obj.gradient_calculator.compute_single_shot_gradient(ishot, forward_wavefield);
    
    % 累加
    fields = fieldnames(total_gradient);
    for i = 1:length(fields)
        total_gradient.(fields{i}) = total_gradient.(fields{i}) + shot_gradient.(fields{i});
    end
    total_misfit = total_misfit + shot_misfit;
end
```

### VTI梯度计算（使用MEX加速）
```matlab
function gradient = compute_vti_gradient(obj, forward_wavefield, adjoint_wavefield)
    % 获取时间步长和其他参数
    dt = obj.adjoint_solver.syn_params.DELTAT;
    params = obj.adjoint_solver.syn_params;
    
    % 调用MEX函数计算梯度
    gradient = compute_vti_gradient_omp(forward_wavefield, adjoint_wavefield, dt, params);
end
```

## 配置和调优指南

### 关键参数设置
1. **优化参数**:
   - `max_iterations`: 最大迭代次数
   - `step_length`: 初始步长
   - `step_length_decay`: 步长衰减因子
   - `max_line_search`: 最大线搜索次数

2. **模型约束**:
   - `c11_min/max`: c11参数范围
   - `c13_min/max`: c13参数范围
   - `c33_min/max`: c33参数范围
   - `c44_min/max`: c44参数范围
   - `rho_min/max`: 密度范围
   - `water_layer`: 水层深度（用于梯度掩码）

3. **输出控制**:
   - `save_to_disk`: 是否保存到磁盘
   - `save_interval`: 保存间隔（迭代次数）
   - `save_shot_gradient`: 是否保存单炮梯度

### 性能优化
1. **内存管理**:
   - 使用`clear`及时清理大型波场数据
   - 使用`try-finally`块确保资源释放
   - 避免不必要的数据复制

2. **计算加速**:
   - 核心梯度计算使用MEX函数实现
   - 利用OpenMP实现多线程并行
   - 考虑SIMD指令集优化

### 调试指南
1. **梯度验证**:
   - 使用`test_Gradient.m`验证梯度计算
   - 检查梯度的数值范围是否合理
   - 监测异常体周围的梯度分布

2. **收敛监测**:
   - 使用`plot_convergence()`查看收敛曲线
   - 检查目标函数值是否单调下降
   - 监测步长变化情况 