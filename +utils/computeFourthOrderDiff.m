%% 四阶中心差分计算空间导数
% 功能：计算二维场的空间导数，内部使用四阶中心差分，边界使用二阶差分
% 
% 说明：
%   1. 主要功能：
%      - 计算x方向空间导数
%      - 计算y方向空间导数
%      - 边界特殊处理
%   2. 计算方法：
%      - 内部点：四阶中心差分
%      - 边界点：二阶前向/后向差分
%      - 次边界点：二阶中心差分
%
% 使用方法：
%   [dx, dy] = utils.computeFourthOrderDiff(field, deltax, deltay)
%
% 输入参数：
%   - field: 输入场 (2D矩阵)
%   - deltax: x方向网格间距
%   - deltay: y方向网格间距
%
% 输出参数：
%   - dx: x方向导数
%   - dy: y方向导数
%
% 依赖项：
%   - 无
%
% 注意事项：
%   - 输入场必须是二维矩阵
%   - 网格间距必须为正数
%   - 边界处理使用较低阶差分以保持稳定性
%
% 作者：StarrMoonn
% 日期：2025-01-16
%

function [dx, dy] = computeFourthOrderDiff(field, deltax, deltay)
    % 四阶中心差分计算空间导数
    %
    % 输入参数:
    %   field: 输入场 (2D矩阵)
    %   deltax: x方向网格间距
    %   deltay: y方向网格间距
    %
    % 输出参数:
    %   dx: x方向导数
    %   dy: y方向导数
    %
    % 说明：
    %   - 内部点使用四阶中心差分
    %   - 边界点使用二阶差分
    %   - 适用于地震波场正演和反演计算
    
    % 获取场的大小
    [nx, ny] = size(field);
    
    % 初始化导数数组
    dx = zeros(nx, ny);
    dy = zeros(nx, ny);
    
    % 四阶中心差分系数（修正系数）
    c1 = -8/12;   % 相邻点系数
    c2 = 1/12;    % 间隔一个点的系数
    
    % 计算x方向导数（四阶中心差分）
    for i = 3:nx-2
        dx(i,:) = (c2*(field(i-2,:) - field(i+2,:)) + ...
                   c1*(field(i-1,:) - field(i+1,:))) / deltax;
    end
    
    % 计算y方向导数（四阶中心差分）
    for j = 3:ny-2
        dy(:,j) = (c2*(field(:,j-2) - field(:,j+2)) + ...
                   c1*(field(:,j-1) - field(:,j+1))) / deltay;
    end
    
    % 处理边界（使用二阶差分）
    % x方向边界
    dx(1,:) = (-3*field(1,:) + 4*field(2,:) - field(3,:)) / (2*deltax);  % 二阶前向差分
    dx(2,:) = (field(3,:) - field(1,:)) / (2*deltax);  % 二阶中心差分
    dx(nx-1,:) = (field(nx,:) - field(nx-2,:)) / (2*deltax);  % 二阶中心差分
    dx(nx,:) = (field(nx-2,:) - 4*field(nx-1,:) + 3*field(nx,:)) / (2*deltax);  % 二阶后向差分
    
    % y方向边界
    dy(:,1) = (-3*field(:,1) + 4*field(:,2) - field(:,3)) / (2*deltay);  % 二阶前向差分
    dy(:,2) = (field(:,3) - field(:,1)) / (2*deltay);  % 二阶中心差分
    dy(:,ny-1) = (field(:,ny) - field(:,ny-2)) / (2*deltay);  % 二阶中心差分
    dy(:,ny) = (field(:,ny-2) - 4*field(:,ny-1) + 3*field(:,ny)) / (2*deltay);  % 二阶后向差分
end 