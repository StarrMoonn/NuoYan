%% 弹性参数数据结构转换工具
% 功能：实现VTI介质弹性参数在结构体和数组形式之间的转换
% 
% 说明：
%   1. 主要功能：
%      - 将弹性参数结构体转换为一维数组（struct2array）
%      - 将一维数组重构为弹性参数结构体（array2struct）
%   
%   2. 支持的弹性参数：
%      - c11：VTI介质刚度系数
%      - c13：VTI介质刚度系数
%      - c33：VTI介质刚度系数
%      - c44：VTI介质刚度系数
%      - rho：密度
%
% 依赖关系：
%   - 被BBOptimizer调用用于：
%     * BB迭代过程中的梯度计算
%     * 步长安全检查
%     * 收敛性判断
%   - 用于优化算法中的向量运算
%
% 输入/输出：
%   struct2array:
%     输入：包含五个弹性参数的结构体
%     输出：连接后的一维数组
%   
%   array2struct:
%     输入：一维数组，nx（网格x维度），ny（网格y维度）
%     输出：包含五个弹性参数的结构体
%
% 作者：StarrMoonn
% 日期：2025-01-10
%
classdef elastic_params_converter
    methods(Static)
        function array = struct2array(elastic_struct)
            % 将弹性参数结构体转换为数组
            fields = {'c11', 'c13', 'c33', 'c44', 'rho'};
            arrays = cellfun(@(f) elastic_struct.(f)(:), fields, 'UniformOutput', false);
            array = vertcat(arrays{:});
        end
        
        function elastic_struct = array2struct(array, nx, ny)
            % 将数组转换回弹性参数结构体
            elastic_struct = struct();
            fields = {'c11', 'c13', 'c33', 'c44', 'rho'};
            n = nx * ny;
            for i = 1:length(fields)
                elastic_struct.(fields{i}) = reshape(array((i-1)*n+1:i*n), nx, ny);
            end
        end
    end
end 