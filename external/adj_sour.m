%% adj_sour - 基于包络差的FWI伴随源构造函数
%
% 功能说明：
%   计算基于包络差的伴随源，用于改进传统FWI中的伴随源构造方法。
%   通过使用包络差代替波形差，可以降低FWI对初始模型的依赖性，提高反演稳定性。
%
% 输入参数：
%   seistureE - 观测数据的包络 [ntime x nreceivers x nshots]
%   seis0     - 正演模拟的地震数据 [ntime x nreceivers]
%   ishot     - 当前炮号
%   dt        - 采样时间间隔
%
% 输出参数：
%   a         - 构造的伴随源
%
% 使用说明：
%   1. 该函数替代传统FWI中的波形差伴随源
%   2. 输出的伴随源直接注入到伴随波动方程中
%   3. 其他FWI框架（如梯度计算、参数更新等）保持不变
%
% 优势：
%   1. 降低目标函数的非线性程度
%   2. 减少周期跳跃问题
%   3. 对初始模型精度要求较低
%
% 关于包络计算：
%   MATLAB的hilbert函数计算解析信号
%   z = hilbert(x)  
%   z = a + bi (解析信号，复数)
%   |z| = sqrt(a² + b²) 就是包络
% 所以包络计算就是：
%   seisv0E = abs(hilbert(seisv0))  % abs求模得到包络




function a=adj_sour(seistureE,seis0,ishot,dt)
seis20=hilbert(seis0);
seis0E=sqrt(real(seis20).^2+imag(seis20).^2);
b=(seis0E-seistureE(:,:,ishot))./(seis0E+0.01).*imag(seis20);
a=seis0.*(seis0E-seistureE(:,:,ishot))./(seis0E+0.01)-imag(hilbert(b));

% b=(seis0E.^2-seistureE(:,:,ishot).^2).*imag(seis20);
% a=seis0.*(seis0E.^2-seistureE(:,:,ishot).^2)-imag(hilbert(b));
%  a=bp_filter(a2,dt,0,0,15,20);
% [a] =  lpfilter(a2,dt,7);
% [bh,ah]=butter(3,10*dt*2,'low');  
% a=filter(bh,ah,a2);
end