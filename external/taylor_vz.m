%% taylor_vz - 计算任意阶中心差分系数
%   使用Fornberg方法计算任意阶数的中心差分系数
%
% 输入参数:
%   Z - 网格点位置向量，例如[-4:4]表示8阶中心差分的9个点
%
% 输出参数:
%   c - 差分系数向量
%
% 参考文献:
%   Fornberg, B. (1988). "Generation of Finite Difference Formulas on 
%   Arbitrarily Spaced Grids". Mathematics of Computation, 51(184), 699-706.
%
% 示例:
%   Z = -4:4;  % 8阶中心差分
%   c = taylor_vz(Z);  % 返回9个差分系数
%
% 注意:
%   对于二阶导数的8阶精度差分，系数应为：
%   [-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]
%

function[c]=taylor_vz(Z)

N=(length(Z)-1)/2;

for ii=1:2*N
    A(:,ii)=[Z(1:N).^ii/factorial(ii) Z(N+2:end).^(ii)/factorial(ii)]';
end
B=inv(A);
c(1:N)=B(2,1:N);
c(N+2:2*N+1)=B(2,N+1:2*N);
c(N+1)=-sum(B(2,:));

    c(N+2:end) = -c(N:-1:1);  % 对称部分
end