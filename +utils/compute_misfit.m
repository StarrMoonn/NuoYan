%% 全波形反演误差函数计算
% 功能：计算观测数据和合成数据之间的L2范数误差
% 
% 说明：
%   1. 读取观测数据和合成数据
%   2. 计算两组数据之间的L2范数误差
%   3. 支持多炮数据的误差累加
%
% 输入：
%   - obs_data: 观测数据 {nshot} cell数组，每个元素是 [nstep x nrec] 矩阵
%   - syn_data: 合成数据 {nshot} cell数组，每个元素是 [nstep x nrec] 矩阵
%
% 输出：
%   - misfit: 总误差值
%   - misfit_per_shot: 每炮的误差值数组
%
% 作者：StarrMoonn
% 日期：2025-01-02
%

function [misfit, misfit_per_shot] = compute_misfit(obs_vx, obs_vy, syn_vx, syn_vy)
    % 获取炮数
    nshot = length(obs_vx);
    misfit_per_shot = zeros(nshot, 1);
    
    % 逐炮计算误差
    for ishot = 1:nshot
        % 计算水平分量误差
        diff_vx = obs_vx{ishot} - syn_vx{ishot};
        error_vx = sum(sum(diff_vx.^2));
        
        % 计算垂直分量误差
        diff_vy = obs_vy{ishot} - syn_vy{ishot};
        error_vy = sum(sum(diff_vy.^2));
        
        % 合并两个分量的误差
        misfit_per_shot(ishot) = 0.5 * (error_vx + error_vy);
    end
    
    % 计算总误差
    misfit = sum(misfit_per_shot);
    
    % 输出信息
    fprintf('\n=== 误差函数计算结果 ===\n');
    fprintf('总误差: %e\n', misfit);
    for ishot = 1:nshot
        fprintf('第 %d 炮误差: %e\n', ishot, misfit_per_shot(ishot));
    end
end 