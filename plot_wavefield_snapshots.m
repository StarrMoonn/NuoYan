%% 波场快照可视化工具
% 功能：将波场数据转换为PNG格式的可视化图像
% 
% 说明：
%   1. 读取JSON配置文件中的模型参数
%   2. 加载波场数据（vx和vy分量）
%   3. 生成彩色波场快照图像
%   4. 在图像中标注：
%      * 震源位置（橙色十字）
%      * PML边界（橙黄色）
%      * 检波器位置（绿色点）
%
% 输入：
%   - JSON参数文件：包含模型尺寸、PML设置等参数
%   - 波场数据文件：wavefield_XXXXXX.mat，包含vx_data和vy_data
%   - 输出目录：保存PNG图像的路径
%
% 输出：
%   - PNG格式的波场快照图像
%   - 文件保存在以炮号命名的子目录下
%   - 文件命名格式：wavefield_XXXXXX.png
%
% 图像特征：
%   - 红色：正值波场
%   - 蓝色：负值波场
%   - 白色：低于阈值的区域
%   - 特殊标记：
%     * 震源：橙色十字
%     * PML边界：橙黄色线
%     * 检波器：绿色点
%
% 作者：StarrMoonn
% 日期：2025-01-02
%

function plot_wavefield_snapshots(json_path, data_path, output_dir)
    % 从JSON文件读取参数并绘制波场快照
    % 参数:
    %   json_path: JSON参数文件路径 (可以是params1.json, params2.json等)
    %   data_path: 波场数据.mat文件所在路径
    %   output_dir: 图像输出路径
    % 输出：
    %   - PNG格式的波场快照图像
    
    % 读取JSON文件
    try
        fid = fopen(json_path, 'r');
        raw = fread(fid, inf);
        str = char(raw');
        fclose(fid);
        params = jsondecode(str);
        
        % 检查必要参数是否存在
        required_fields = {'NX', 'NY', 'NPOINTS_PML', 'PML_XMIN', 'PML_XMAX', ...
                         'PML_YMIN', 'PML_YMAX'};
        for i = 1:length(required_fields)
            if ~isfield(params, required_fields{i})
                error('缺少必要参数: %s', required_fields{i});
            end
        end
    catch ME
        error('读取JSON文件失败: %s', ME.message);
    end
    
    % 创建输出目录
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % 可视化参数设置
    POWER_DISPLAY = 0.30;
    cutvect = 0.01;
    WHITE_BACKGROUND = true;
    width_cross = 5;
    thickness_cross = 1;
    
    % 获取所有波场数据文件
    files = dir(fullfile(data_path, 'wavefield_*.mat'));
    
    % 遍历所有数据文件
    for file_idx = 1:length(files)
        % 加载波场数据
        data = load(fullfile(data_path, files(file_idx).name));
        it = sscanf(files(file_idx).name, 'wavefield_%d.mat');
        
        % 处理Vx和Vy分量
        for field_number = 1:2
            % 选择波场分量
            if field_number == 1
                image_data_2D = data.vx_data;
                field_name = 'Vx';
            else
                image_data_2D = data.vy_data;
                field_name = 'Vy';
            end
            
            % 生成输出文件名
            fig_name = sprintf('image%06d_%s.png', it, field_name);
            fig_name = fullfile(output_dir, fig_name);
            
            % 计算最大振幅
            max_amplitude = max(max(abs(image_data_2D)));
            
            % 创建RGB图像数组
            img = zeros(params.NY, params.NX, 3);
            
            % 计算检波器位置（根据first_rec_x/y和NREC计算）
            if isfield(params, 'first_rec_x') && isfield(params, 'first_rec_y') && isfield(params, 'NREC')
                rec_x = round(params.first_rec_x/params.DELTAX) + (0:params.NREC-1);
                rec_y = round(params.first_rec_y/params.DELTAY) * ones(1, params.NREC);
            else
                rec_x = [];
                rec_y = [];
            end
            
            % 计算震源位置
            isource = round(params.NX/2);  % 默认在中心
            jsource = round(params.NY/2);
            
            % 逐点填充图像
            for iy = params.NY:-1:1
                for ix = 1:params.NX
                    % 归一化数值
                    normalized_value = image_data_2D(ix,iy) / max_amplitude;
                    normalized_value = max(min(normalized_value, 1.0), -1.0);
                    
                    % 绘制震源位置（橙色十字）
                    if ((ix >= isource - width_cross && ix <= isource + width_cross && ...
                         iy >= jsource - thickness_cross && iy <= jsource + thickness_cross) || ...
                        (ix >= isource - thickness_cross && ix <= isource + thickness_cross && ...
                         iy >= jsource - width_cross && iy <= jsource + width_cross))
                        img(iy,ix,:) = [1.0, 0.616, 0.0];
                        
                    % 绘制边框（黑色）
                    elseif ix <= 1 || ix >= params.NX-2 || iy <= 1 || iy >= params.NY-2
                        img(iy,ix,:) = [0.0, 0.0, 0.0];
                        
                    % 绘制PML边界（橙黄色）
                    elseif (params.PML_XMIN && ix == params.NPOINTS_PML) || ...
                           (params.PML_XMAX && ix == params.NX - params.NPOINTS_PML) || ...
                           (params.PML_YMIN && iy == params.NPOINTS_PML) || ...
                           (params.PML_YMAX && iy == params.NY - params.NPOINTS_PML)
                        img(iy,ix,:) = [1.0, 0.588, 0.0];
                        
                    % 绘制检波器位置（绿色）
                    elseif ~isempty(rec_x) && any(ix == rec_x & iy == rec_y)
                        img(iy,ix,:) = [0.0, 1.0, 0.0];
                        
                    % 处理低于阈值的点
                    elseif abs(image_data_2D(ix,iy)) <= max_amplitude * cutvect
                        if WHITE_BACKGROUND
                            img(iy,ix,:) = [1.0, 1.0, 1.0];
                        else
                            img(iy,ix,:) = [0.0, 0.0, 0.0];
                        end
                        
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
            fprintf('已保存图像: %s\n', fig_name);
        end
    end
end


