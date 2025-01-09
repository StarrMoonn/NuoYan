function create_reverse_wavefield_gif(input_dir, output_filename, delay_time)
    % 获取所有PNG文件
    files = dir(fullfile(input_dir, '*.png'));
    fprintf('找到 %d 个PNG文件\n', length(files));
    
    % 创建colormap
    cmap = jet(256);  % 使用256色的jet colormap
    
    % 反向遍历所有文件
    for i = length(files):-1:1  % 这里改为反向遍历
        % 读取PNG图像
        [img, ~] = imread(fullfile(input_dir, files(i).name));
        
        % 如果是RGB图像，转换为索引图像
        if size(img, 3) == 3
            [img_indexed, cmap] = rgb2ind(img, 256);
        else
            img_indexed = img;
        end
        
        % 如果是第一帧（原来的最后一帧）
        if i == length(files)  % 这里改为检查是否是最后一个索引
            imwrite(img_indexed, cmap, output_filename, 'gif', ...
                   'LoopCount', Inf, ...
                   'DelayTime', delay_time);
        else
            % 添加后续帧
            imwrite(img_indexed, cmap, output_filename, 'gif', ...
                   'WriteMode', 'append', ...
                   'DelayTime', delay_time);
        end
        
        fprintf('处理第 %d/%d 帧\n', length(files)-i+1, length(files));
    end
    
    fprintf('反向GIF创建完成: %s\n', output_filename);
end 