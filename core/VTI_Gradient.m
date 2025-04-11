%% VTI介质全波形反演梯度计算模块
% 功能：计算VTI介质中全波形反演的梯度
% 
% 说明：
%   1. 主要功能：
%      - 计算单炮梯度
%      - 弹性参数(c11,c13,c33,c44)和密度的梯度计算
%   2. 计算方法：
%      - 基于速度场的梯度计算（通过MEX函数实现）
%      - 二阶中心差分计算空间导数
%      - 时间导数采用中心差分（边界使用单侧差分）
%
% 类属性：
%   - adjoint_solver: 伴随波场求解器实例
%   - gradient_output_dir: 梯度输出目录
%   - NSTEP: 时间步数
%
% 主要方法：
%   1. 构造函数：
%      - VTI_Gradient(params): 初始化梯度计算器
%   2. 梯度计算：
%      - compute_single_shot_gradient: 计算单炮梯度
%      - compute_vti_gradient: 基于速度场计算梯度（调用MEX）
%   3. 辅助功能：
%      - save_gradient: 保存梯度结果
%
% 梯度计算公式：
%   速度场方法：
%   - c11梯度：-∂vx/∂x * ∂v†x/∂x
%   - c13梯度：-(∂v†x/∂x * ∂vy/∂y + ∂v†y/∂y * ∂vx/∂x)
%   - c33梯度：-∂vy/∂y * ∂v†y/∂y
%   - c44梯度：-(∂vx/∂y + ∂vy/∂x) * (∂v†x/∂y + ∂v†y/∂x)
%   - ρ梯度：-v†i * ∂²vi/∂t²
%
% 优化说明：
%   1. MEX加速：
%      - 使用C++编写核心计算代码
%      - OpenMP并行优化
%      - SIMD向量化优化
%   2. 内存管理：
%      - 自动清理临时变量
%      - 分步释放大型波场数据
%
% 依赖项：
%   - VTI_Adjoint类：提供伴随波场计算
%   - compute_vti_gradient_omp.cpp：C++实现的梯度计算
%
% 输出说明：
%   - 梯度结构体包含：c11, c13, c33, c44, rho
%   - 自动保存到指定输出目录
%   - 支持单炮梯度的独立保存
%
% 注意事项：
%   1. 内存管理：
%      - 波场数据量较大，注意及时清理
%      - 使用try-finally确保资源释放
%   2. 性能优化：
%      - 核心计算已移至C++实现，提高计算效率
%
% 作者：StarrMoonn
% 日期：2025-04-10
% 
classdef VTI_Gradient < handle
    properties
        adjoint_solver           % 伴随波场求解器实例
        gradient_output_dir      % 梯度输出目录
        NSTEP                    % 时间步数
        save_shot_gradient       % 是否保存单炮梯度
    end
     
    methods
        % 构造函数
        function obj = VTI_Gradient(params)
            % 创建伴随波场求解器实例
            obj.adjoint_solver = VTI_Adjoint(params);
            
            % 设置输出目录
            current_dir = fileparts(fileparts(mfilename('fullpath')));
            obj.gradient_output_dir = fullfile(current_dir, 'data', 'output', 'gradient');
            
            % 从合成数据参数中获取时间步数
            obj.NSTEP = obj.adjoint_solver.syn_params.NSTEP;
            
            if ~exist(obj.gradient_output_dir, 'dir')
                mkdir(obj.gradient_output_dir);
            end
            
            % 是否保存单炮梯度，默认为false
            if isfield(params, 'save_shot_gradient')
                obj.save_shot_gradient = params.save_shot_gradient;
            else
                obj.save_shot_gradient = false;
            end
        end
        
        % 计算单炮梯度
        function [gradient, misfit] = compute_single_shot_gradient(obj, ishot, forward_wavefield)
            fprintf('\n=== 开始计算第 %d 炮梯度 ===\n', ishot);

            try
                % 先计算残差和目标函数值
                misfit = obj.adjoint_solver.compute_residuals_single_shot(ishot);
                fprintf('第 %d 炮目标函数值: %e\n', ishot, misfit);
                
                % 再计算伴随波场
                adjoint_wavefield = obj.adjoint_solver.compute_adjoint_wavefield_single_shot(ishot);
                
                % 使用正演波场和伴随波场计算梯度
                gradient = obj.compute_vti_gradient(forward_wavefield, adjoint_wavefield);
                
                % 可选地保存单炮梯度
                if obj.save_shot_gradient
                    obj.save_gradient(gradient, ishot);
                end
                
            catch ME
                fprintf('计算梯度时发生错误: %s\n', ME.message);
                rethrow(ME);
                
            finally
                % 清理内存
                if exist('adjoint_wavefield', 'var')
                    clear adjoint_wavefield;
                end
            end
            
            fprintf('\n=== 炮号 %d 梯度计算完成 ===\n', ishot);
        end
        
        % 计算VTI介质梯度
        function gradient = compute_vti_gradient(obj, forward_wavefield, adjoint_wavefield)
            % 获取时间步长和其他参数
            dt = obj.adjoint_solver.syn_params.DELTAT;
            params = obj.adjoint_solver.syn_params;
            
            % 调用 MEX 函数计算梯度
            try
                gradient = compute_vti_gradient_omp(forward_wavefield, adjoint_wavefield, dt, params);
            catch ME
                fprintf('MEX函数调用失败，错误信息：%s\n', ME.message);
                rethrow(ME);
            end
        end
        
        % 保存梯度
        function save_gradient(obj, gradient, ishot)
            % 验证输入
            if ~isstruct(gradient) || ~all(isfield(gradient, {'c11','c13','c33','c44','rho'}))
                error('梯度必须包含所有必要字段：c11, c13, c33, c44, rho');
            end
            
            % 验证所有分量维度一致性
            [nx_c11, ny_c11] = size(gradient.c11);
            fields = {'c13', 'c33', 'c44', 'rho'};
            for field = fields
                [nx, ny] = size(gradient.(field{1}));
                if nx ~= nx_c11 || ny ~= ny_c11
                    error('梯度分量维度不一致：%s [%d, %d] vs c11 [%d, %d]', ...
                          field{1}, nx, ny, nx_c11, ny_c11);
                end
            end
            
            fprintf('\n=== 梯度维度检查 ===\n');
            fprintf('梯度维度: [%d, %d]\n', nx_c11, ny_c11);
            
            % 构造文件名
            filename = sprintf('gradient_shot_%d.mat', ishot);
            filepath = fullfile(obj.gradient_output_dir, filename);
            
            % 保存梯度
            save(filepath, 'gradient');
            fprintf('梯度已保存到: %s\n', filepath);
        end
    end
end 