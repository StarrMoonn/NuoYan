%% control_v - 速度模型约束函数
%
% 功能说明：
%   控制速度模型的值在指定范围内，防止速度值在反演过程中出现非物理的值。
%   考虑水层速度不变的情况，只对水层以下的速度进行约束。
%
% 输入参数：
%   v     - 输入速度模型 [nz x nx]
%   water - 水层厚度（网格点数）
%   vmin  - 最小允许速度值
%   vmax  - 最大允许速度值
%
% 输出参数：
%   vel   - 约束后的速度模型 [nz x nx]
%
% 速度约束示意图：
%   ┌──────────────────┐ ← i=1
%   │     水层速度      │ ← 不做约束
%   ├──────────────────┤ ← i=water+1
%   │                  │
%   │   地下速度约束    │ ← vmin ≤ v ≤ vmax
%   │                  │
%   └──────────────────┘ ← i=nz
%
% 注意事项：
%   1. 水层速度保持不变
%   2. 只对水层以下的速度进行约束
%   3. 约束范围应该根据实际地质情况设置
%
% 使用示例：
%   vel = control_v(v, 10, 1500, 4500);
%   % 水层深度10个网格点
%   % 最小速度1500m/s
%   % 最大速度4500m/s
%
% 在FWI中的应用：
%   - 每次速度更新后使用
%   - 保证速度模型物理合理性
%   - 提高反演稳定性
%

function vel=control_v(v,water,vmin,vmax)
% 获取模型尺寸
[nz,nx]=size(v);
vel=v;

% 限制最小速度
for i=(water+1):nz
    for j=1:nx
     if vel(i,j)<vmin
        vel(i,j)=vmin;
     end
    end
end

% 限制最大速度
for i=(water+1):nz
    for j=1:nx
     if vel(i,j)>vmax
        vel(i,j)=vmax;
     end
    end
end

end