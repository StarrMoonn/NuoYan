%% JSON参数文件加载函数
% 功能：加载并验证JSON格式的模型参数文件
% 
% 说明：
%   1. 读取指定的JSON配置文件
%   2. 验证必要参数的存在性
%   3. 加载材料参数的mat文件
%   4. 返回完整的参数结构体
%
% 输入参数：
%   - json_file: JSON文件的完整路径
%
% 输出参数：
%   - params: 包含所有参数的结构体，包括：
%     * 网格参数 (NX, NY, DELTAX, DELTAY)
%     * 时间步进参数 (NSTEP, DELTAT)
%     * 震源参数 (first_shot_i/j, shot_di/j, NSHOT)
%     * 材料参数数组 (c11, c13, c33, c44, rho)
%     * 其他配置参数
%
% 注意事项：
%   1. JSON文件必须包含所有必要参数
%   2. 材料参数可以是mat文件路径
%   3. 确保mat文件路径正确
%
% 示例：
%   params = utils.load_json_params('data/params/case1/params.json');
%
% 依赖：
%   - jsondecode (MATLAB内置)
%   - fieldnames (MATLAB内置)
%
% 作者：StarrMoonn
% 日期：2025-01-02
%

function params = load_json_params(json_file)
    % 检查文件是否存在
    if ~exist(json_file, 'file')
        error('找不到JSON文件: %s', json_file);
    end
    
    % 读取JSON文件
    try
        fid = fopen(json_file, 'r');
        if fid == -1
            error('无法打开JSON文件: %s', json_file);
        end
        raw = fread(fid, inf);
        str = char(raw');
        fclose(fid);
        params = jsondecode(str);
    catch ME
        if exist('fid', 'var') && fid ~= -1
            fclose(fid);
        end
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
    
    % 处理材料参数数组
    material_params = {'c11', 'c13', 'c33', 'c44', 'rho'};
    for i = 1:length(material_params)
        param_name = material_params{i};
        if isstruct(params.(param_name)) && isfield(params.(param_name), 'type')
            if strcmp(params.(param_name).type, 'mat')
                try
                    data = load(params.(param_name).file);
                    field_names = fieldnames(data);
                    params.(param_name) = data.(field_names{1});
                catch ME
                    error('无法加载%s文件: %s', params.(param_name).file, ME.message);
                end
            end
        end
    end
end 