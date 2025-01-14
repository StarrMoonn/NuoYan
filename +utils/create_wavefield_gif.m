function create_wavefield_gif(input_dir, output_filename, delay_time)
    %% 波场GIF生成工具
    % 功能：将指定目录下的PNG图像序列合成为GIF动画
    % 
    % 说明：
    %   1. 读取指定目录下的所有PNG图像文件
    %   2. 按照文件名顺序逐帧合成GIF动画
    %   3. 支持RGB图像自动转换为索引图像
    %
    % 输入：
    %   - input_dir: 包含PNG图像序列的目录路径
    %   - output_filename: 输出GIF文件的完整路径
    %   - delay_time: 动画帧间延迟时间（秒）
    %
    % 输出：
    %   - 无（直接生成GIF文件）
    %
    % 作者：StarrMoonn
    % 日期：2025-01-014
    %
    
    files = dir(fullfile(input_dir, '*.png'));
    fprintf('找到 %d 个PNG文件\n', length(files));
    
    % 创建colormap
    cmap = jet(256);  % 使用256色的jet colormap
    
    % 遍历所有文件
    for i = 1:length(files)
        % 读取PNG图像
        [img, ~] = imread(fullfile(input_dir, files(i).name));
        
        % 如果是RGB图像，转换为索引图像
        if size(img, 3) == 3
            [img_indexed, cmap] = rgb2ind(img, 256);
        else
            img_indexed = img;
        end
        
        % 如果是第一帧
        if i == 1
            imwrite(img_indexed, cmap, output_filename, 'gif', ...
                   'LoopCount', Inf, ...
                   'DelayTime', delay_time);
        else
            % 添加后续帧
            imwrite(img_indexed, cmap, output_filename, 'gif', ...
                   'WriteMode', 'append', ...
                   'DelayTime', delay_time);
        end
        
        fprintf('处理第 %d/%d 帧\n', i, length(files));
    end
    
    fprintf('GIF创建完成: %s\n', output_filename);
end 