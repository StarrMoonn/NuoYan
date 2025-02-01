%% cross_resid - 伴随源-波形互相关残差计算函数（标准版）
%
% 功能说明：
%   计算观测数据和模拟数据之间的互相关残差，用于反传计算伴随波场。
%   通过归一化处理和互相关运算，减少振幅差异的影响，主要关注波形相似性。
%
% 输入参数：
%   seisv0   - 模拟地震记录 [nt x nx]
%   seisture - 观测地震记录 [nt x nx]
%   其中：nt为采样点数，nx为道数
%
% 输出参数：
%   resid    - 互相关残差 [nt x nx]
%
% 计算步骤：
%   1. 对每道数据进行归一化处理
%   2. 计算互相关系数
%   3. 构建残差
%
% 计算公式：
%   1. 归一化：trace = trace/||trace||₂
%   2. 互相关：b = sum(seisv0.*seisture)
%   3. 残差：resid = (c-seisture)/a
%      其中：c是基于互相关系数重构的波形
%
% 注意事项：
%   1. 输入数据需要有相同的维度
%   2. 处理了零能量道的特殊情况
%   3. 最终残差经过了整体能量归一化
%
% 使用示例：
%   resid = cross_resid(synthetic_data, observed_data);
%
% 具体来说，都是伴随源计算函数：
% cross_resid.m:
% 正演模拟 → 包络差计算 → 伴随源构造 → 伴随波场计算 → 梯度计算
%                            
% adj_sour.m:
% 正演模拟 → 包络差计算 → 伴随源构造 → 伴随波场计算 → 梯度计算
%% FWI迭代流程：
%
% 1. 伴随源和梯度计算：
%    正演波场 ──→ 包络计算
%                   │
%    观测数据 ──→ 包络计算
%                   │
%                   ↓
%    adj_sour.m (包络差伴随源)
%           │
%           ↓
%    伴随波场计算
%           │
%           ↓
%       梯度计算
%           │
%           ↓
%       模型更新

function resid=cross_resid(seisv0,seisture)
% 获取道数
nx=size(seisv0,2);

% 计算模拟数据的整体能量
a=norm(seisv0,2);

% 对每道数据进行归一化
for j=1:nx
    % 只处理非零能量道
    if norm(seisv0(:,j),2)~=0 && norm(seisture(:,j),2)~=0
        seisv0(:,j)=seisv0(:,j)/norm(seisv0(:,j),2);
        seisture(:,j)=seisture(:,j)/norm(seisture(:,j),2);
    end
end

% 计算互相关系数
b=sum(seisv0.*seisture,1);

% 获取数据维度
[m,n]=size(seisv0);
c=zeros(m,n);

% 构建基于互相关的波形
for j=1:nx
   c(:,j)=seisv0(:,j)*b(j); 
end

% 计算最终残差
resid=(c-seisture)/a;

end