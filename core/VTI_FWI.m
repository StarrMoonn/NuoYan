%% VTI介质全波形反演主控制模块
% 功能：实现VTI介质的全波形反演优化迭代过程
% 
% 说明：
%   1. 主要功能：
%      - 执行完整的FWI迭代优化过程
%      - 支持多种优化算法（梯度下降法、BB法和L-BFGS法）
%      - 自动步长计算和收敛控制
%
% 输入参数：
%   params结构体包含：
%   - project_root：项目根目录
%   - obs_json_file：观测数据参数文件
%   - syn_json_file：合成数据参数文件
%   - fwi：FWI相关参数
%     * max_iterations：最大迭代次数
%     * tolerance：收敛容差
%     * optimization_method：优化方法选择
%     * lbfgs_memory_length：L-BFGS存储长度（可选）
%
% 输出：
%   - 迭代梯度、目标函数值和收敛曲线
%
% 作者：StarrMoonn
% 日期：2025-01-10
%
classdef VTI_FWI < handle  
    properties
        optimizer    % 优化器实例
    end
    
    methods
        function obj = VTI_FWI(params)
            % 构造函数：根据选择创建相应的优化器
            switch params.optimization_method
                case 'gradient_descent'
                    obj.optimizer = GradientDescentOptimizer(params);
                case 'BB'
                    obj.optimizer = BBOptimizer(params);
                case 'LBFGS'
                    obj.optimizer = LBFGSOptimizer(params);
                otherwise
                    error('未知的优化方法: %s', params.optimization_method);
            end
        end
        
        function run(obj)
            % 运行选定的优化算法
            obj.optimizer.run();
        end
    end
end 