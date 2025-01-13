%% 伴随波场快照可视化工具
% 功能：将伴随波场数据转换为PNG格式的可视化图像
% 
% 说明：
%   1. 读取JSON配置文件中的模型参数
%   2. 加载伴随波场数据（vx或vy分量）
%   3. 生成彩色波场快照图像
%   4. 在图像中标注：
%      * 检波器位置（橙色十字）
%      * PML边界（橙黄色）
%      * 边框（黑色）
%
% 输入参数：
%   - json_path：JSON参数文件路径，包含：
%       * NX, NY：模型尺寸
%       * NPOINTS_PML：PML层厚度
%       * PML_XMIN, PML_XMAX, PML_YMIN, PML_YMAX：PML边界标志
%       * NREC：检波器数量
%       * first_rec_x, first_rec_y：初始检波器网格位置
%       * rec_dx, rec_dy：检波器间隔（网格点数）
%   - data_path：波场数据文件路径（adjoint_wavefield_XXXXXX.mat）
%   - output_dir：PNG图像输出目录
%   - component：波场分量选择（'vx'或'vy'）
%
% 输出：
%   - PNG格式的波场快照图像序列
%   - 文件命名格式：adjoint_image_XXXXXX.png
%
% 注意：
%   - 检波器位置使用网格索引表示
%   - 波场值使用红色表示正值，蓝色表示负值
%   - 低于阈值的波场值显示为白色背景
%
% 作者：StarrMoonn
% 日期：2025-01-08
%

function plot_adjoint_snapshots(json_path, data_path, output_dir, component)
    try
        fid = fopen(json_path, 'r');
        raw = fread(fid, inf);
        str = char(raw');
        fclose(fid);
        params = jsondecode(str);
        
        % 检查必要参数是否存在
        required_fields = {'NX', 'NY', 'NPOINTS_PML', 'PML_XMIN', 'PML_XMAX', ...
                         'PML_YMIN', 'PML_YMAX', 'NREC', 'first_rec_x', 'first_rec_y', ...
                         'DELTAX', 'DELTAY', 'rec_dx', 'rec_dy'};
        for i = 1:length(required_fields)
            if ~isfield(params, required_fields{i})
                error('缺少必要参数: %s', required_fields{i});
            end
        end
    catch ME
        error('读取JSON文件失败: %s', ME.message);
    end
    
    % 计算所有检波器位置
    rec_x = zeros(1, params.NREC);
    rec_y = zeros(1, params.NREC);
    for i = 1:params.NREC
        % 使用正确的检波器间隔
        rec_x(i) = params.first_rec_x + (i-1) * params.rec_dx;  % 考虑检波器间隔
        rec_y(i) = params.first_rec_y;                          % y方向保持不变
    end
    
    % 可视化参数设置
    POWER_DISPLAY = 0.30;
    cutvect = 0.01;
    width_cross = 5;      % 十字花的宽度
    thickness_cross = 1;  % 十字花的粗细
    
    % 创建输出目录
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % 获取所有伴随波场数据文件
    files = dir(fullfile(data_path, 'adjoint_wavefield_*.mat'));
    
    % 遍历所有数据文件
    for file_idx = 1:length(files)
        % 加载波场数据
        data = load(fullfile(data_path, files(file_idx).name));
        it = sscanf(files(file_idx).name, 'adjoint_wavefield_%d.mat');
        
        % 根据指定的分量选择数据
        if strcmpi(component, 'vx')
            image_data_2D = data.vx_data;
        else % vy
            image_data_2D = data.vy_data;
        end
        
        % 生成输出文件名
        fig_name = sprintf('adjoint_image_%06d.png', it);  % 移除分量标识，因为已经分目录了
        fig_name = fullfile(output_dir, fig_name);
        
        % 计算最大振幅
        max_amplitude = max(max(abs(image_data_2D)));
        
        % 创建RGB图像数组
        img = zeros(params.NY, params.NX, 3);
        
        % 逐点填充图像
        for iy = params.NY:-1:1
            for ix = 1:params.NX
                % 归一化数值
                normalized_value = image_data_2D(ix,iy) / max_amplitude;
                normalized_value = max(min(normalized_value, 1.0), -1.0);
                
                % 检查是否是检波器位置（用十字花标记）
                is_receiver = false;
                for r = 1:params.NREC
                    if ((ix >= rec_x(r) - width_cross && ix <= rec_x(r) + width_cross && ...
                         iy >= rec_y(r) - thickness_cross && iy <= rec_y(r) + thickness_cross) || ...
                        (ix >= rec_x(r) - thickness_cross && ix <= rec_x(r) + thickness_cross && ...
                         iy >= rec_y(r) - width_cross && iy <= rec_y(r) + width_cross))
                        is_receiver = true;
                        break;
                    end
                end
                
                if is_receiver
                    img(iy,ix,:) = [1.0, 0.616, 0.0];  % 橙色十字花标记检波器位置
                % 绘制边框（黑色）
                elseif ix <= 1 || ix >= params.NX-2 || iy <= 1 || iy >= params.NY-2
                    img(iy,ix,:) = [0.0, 0.0, 0.0];
                    
                % 绘制PML边界（橙黄色）
                elseif (params.PML_XMIN && ix == params.NPOINTS_PML) || ...
                       (params.PML_XMAX && ix == params.NX - params.NPOINTS_PML) || ...
                       (params.PML_YMIN && iy == params.NPOINTS_PML) || ...
                       (params.PML_YMAX && iy == params.NY - params.NPOINTS_PML)
                    img(iy,ix,:) = [1.0, 0.588, 0.0];
                    
                % 处理低于阈值的点
                elseif abs(image_data_2D(ix,iy)) <= max_amplitude * cutvect
                    img(iy,ix,:) = [1.0, 1.0, 1.0];  % 白色背景
                    
                % 处理正常波场值
                else
                    if normalized_value >= 0.0
                        img(iy,ix,:) = [normalized_value^POWER_DISPLAY, 0.0, 0.0];
                    else
                        img(iy,ix,:) = [0.0, 0.0, abs(normalized_value)^POWER_DISPLAY];
                    end
                end
            end
        end
        
        % 保存图像
        imwrite(img, fig_name);
        fprintf('已保存伴随波场图像: %s\n', fig_name);
    end
end 


