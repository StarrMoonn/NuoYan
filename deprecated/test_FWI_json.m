%% 基于JSON配置的逐炮正演程序测试
% 功能：从JSON文件读取参数并测试VTI介质中的多炮正演模拟
% 
% 说明：
%   1. 读取JSON配置文件中的模型参数
%   2. 创建VTI_FWI实例进行逐炮模拟
%   3. 记录每炮的地震记录和波场快照
%
% 输入：
%   - JSON参数文件：包含所有必要的模型参数
%   - 文件路径设置
%
% 输出：
%   - 每炮的地震记录（水平和垂直分量）
%   - 波场快照（可选）
%   - 每炮的计算时间统计
%
% 作者：StarrMoonn
% 日期：2025-01-02
%

clear;
clc;

% 设置项目根目录
project_root = 'E:\Matlab\VTI_project';
cd(project_root);

% 添加项目根目录到路径
addpath(project_root);

% 验证路径是否正确
if ~exist('VTI_FWI', 'class')
    error('无法找到 VTI_FWI 类，请检查路径设置');
end

% 指定JSON文件路径
json_file = fullfile(project_root, 'data', 'params', 'case2', 'params_obs.json');

% 读取JSON文件
try
    fid = fopen(json_file, 'r');
    raw = fread(fid, inf);
    str = char(raw');
    fclose(fid);
    params = jsondecode(str);
catch ME
    error('读取JSON文件失败: %s', ME.message);
end

% 验证必要参数
required_fields = {'NX', 'NY', 'DELTAX', 'DELTAY', 'NSTEP', 'DELTAT', ...
                  'first_shot_i', 'first_shot_j', 'shot_di', 'shot_dj', 'NSHOT'};
                  
for i = 1:length(required_fields)
    if ~isfield(params, required_fields{i})
        error('JSON文件缺少必要参数: %s', required_fields{i});
    end
end

% 打印参数信息
fprintf('\n=== JSON参数文件信息 ===\n');
fprintf('网格尺寸: %d x %d\n', params.NX, params.NY);
fprintf('空间步长: dx=%e, dy=%e\n', params.DELTAX, params.DELTAY);
fprintf('时间步数: %d, 步长: %e\n', params.NSTEP, params.DELTAT);
fprintf('总炮数: %d\n', params.NSHOT);
fprintf('第一炮位置: (%d, %d)\n', params.first_shot_i, params.first_shot_j);
fprintf('炮点间隔: dx=%d, dy=%d\n', params.shot_di, params.shot_dj);

% 处理材料参数数组
material_params = {'c11', 'c13', 'c33', 'c44', 'rho'};
for i = 1:length(material_params)
    param_name = material_params{i};
    if isstruct(params.(param_name)) && isfield(params.(param_name), 'type')
        if strcmp(params.(param_name).type, 'mat')
            try
                % 加载mat文件
                data = load(params.(param_name).file);
                field_names = fieldnames(data);
                params.(param_name) = data.(field_names{1});
                fprintf('成功加载%s数组，大小: [%d, %d]\n', param_name, ...
                    size(params.(param_name), 1), size(params.(param_name), 2));
            catch ME
                error('无法加载%s文件: %s', params.(param_name).file, ME.message);
            end
        end
    end
end

% 创建VTI_FWI实例
fprintf('\n=== 创建VTI_FWI实例并初始化 ===\n');
fwi_model = VTI_FWI(params);

% 运行正演模拟
fwi_model.forward_modeling_all_shots();

fprintf('\n=== 测试完成 ===\n'); 