%% VTI介质全波形反演主控制模块
% 功能：实现VTI介质的全波形反演优化迭代过程
%
% 说明：
%   1. 主要功能：
%      - 执行完整的FWI迭代优化过程
%      - 默认使用梯度下降法
%      - 支持多种优化算法（梯度下降法、BB法和L-BFGS法）
%
% 输入参数：
%   params结构体包含：
%   - optimization_method：优化方法选择
%     * 'gradient_descent'：梯度下降法（默认）
%     * 'BB'：BB算法
%     * 'LBFGS'：L-BFGS算法
%   - 其他优化器所需参数
%
% 作者：StarrMoonn
% 日期：2025-01-10
%
classdef VTI_FWI < handle  
    properties (SetAccess = private)
        optimizer            % 优化器实例（BaseOptimizer的子类实例）
    end
    
    methods
        function obj = VTI_FWI(params)
            % 初始化优化器
            if ~isfield(params, 'optimization')
                params.optimization = 'gradient_descent';  % 默认使用梯度下降法
            end
            
            % 根据指定方法创建具体的优化器实例
            switch params.optimization
                case 'gradient_descent'
                    obj.optimizer = GradientDescentOptimizer(params);
                case 'BB'
                    obj.optimizer = BBOptimizer(params);
                case 'LBFGS'
                    obj.optimizer = LBFGSOptimizer(params);
                otherwise
                    error('未知的优化方法: %s', params.optimization);
            end
        end
        
        function run(obj)
            % 运行FWI优化
            obj.optimizer.run();
        end
    end
end 