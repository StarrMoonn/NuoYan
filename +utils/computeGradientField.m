%% MATLAB的gradient函数计算场的空间导数
%
% 功能：
%   二阶中心差分计算空间导数，使用MATLAB内置的gradient函数
%
% 输入参数：
%   field: 输入场 (2D矩阵)
%   deltax: x方向网格间距 (单位：m)
%   deltay: y方向网格间距 (单位：m)
%
% 输出参数：
%   dx: x方向导数 (与输入场同维度)
%   dy: y方向导数 (与输入场同维度)
%
% 注意事项：
%   - gradient函数返回顺序为[dy, dx]，此函数将其调整为[dx, dy]
%   - 导数已除以网格间距，具有正确的物理单位
%
% 作者：StarrMoonn
% 日期：2025-01-16
%
function [dx, dy] = computeGradientField(field, deltax, deltay)
    % 计算导数并除以实际间距
    [dy, dx] = gradient(field);
    
    % 除以网格间距得到物理单位
    dx = dx / deltax;
    dy = dy / deltay;
end 