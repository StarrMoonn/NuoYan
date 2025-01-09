function compute_wave_propagation_gpu(obj)
    % GPU版本的波场传播计算
    % 检查GPU是否支持双精度
    gpu = gpuDevice();
    if ~gpu.DoubleSupport
        error('当前GPU不支持双精度计算');
    end
    
    try
        % 检查GPU内存是否足够
        [NX, NY] = size(obj.vx);
        required_memory = NX * NY * 8 * 20;  % 估计需要的内存（字节）
        if required_memory > gpu.AvailableMemory
            error('GPU内存不足，需要 %.2f GB，可用 %.2f GB', ...
                  required_memory/1e9, gpu.AvailableMemory/1e9);
        end
        
        % 记录开始时间
        t_start = tic;
        
        % 将数据转移到GPU，使用双精度
        [vx_gpu, vy_gpu, sigmaxx_gpu, sigmayy_gpu, sigmaxy_gpu, ...
         memory_variables_gpu, pml_coeffs_gpu] = prepare_gpu_arrays(obj);
        
        % 常量转换为双精度
        [dx, dy, dt, c11, c13, c33, c44, rho] = prepare_constants(obj);
        
        % 调用CUDA MEX函数计算应力
        [sigmaxx_gpu, sigmayy_gpu, sigmaxy_gpu, ...
         memory_dvx_dx_gpu, memory_dvy_dy_gpu, ...
         memory_dvy_dx_gpu, memory_dvx_dy_gpu] = ...
        compute_stress(vx_gpu, vy_gpu, sigmaxx_gpu, sigmayy_gpu, sigmaxy_gpu, ...
                      memory_variables_gpu, pml_coeffs_gpu, ...
                      dx, dy, dt, c11, c13, c33, c44, NX, NY);
        
        % 调用CUDA MEX函数计算速度
        [vx_gpu, vy_gpu, ...
         memory_dsigmaxx_dx_gpu, memory_dsigmaxy_dy_gpu, ...
         memory_dsigmaxy_dx_gpu, memory_dsigmayy_dy_gpu] = ...
        compute_velocity(sigmaxx_gpu, sigmayy_gpu, sigmaxy_gpu, vx_gpu, vy_gpu, ...
                        memory_variables_gpu, pml_coeffs_gpu, ...
                        dx, dy, dt, rho, NX, NY);
        
        % 将结果传回CPU
        gather_results(obj, vx_gpu, vy_gpu, sigmaxx_gpu, sigmayy_gpu, sigmaxy_gpu, ...
                      memory_variables_gpu);
        
        % 记录计算时间
        compute_time = toc(t_start);
        if compute_time > 0.1
            fprintf('GPU计算用时: %.3f 秒\n', compute_time);
        end
        
    catch ME
        reset(gpu);  % 重置GPU状态
        rethrow(ME);
    end
end

function [vx_gpu, vy_gpu, sigmaxx_gpu, sigmayy_gpu, sigmaxy_gpu, ...
         memory_variables_gpu, pml_coeffs_gpu] = prepare_gpu_arrays(obj)
    % 准备GPU数组
    vx_gpu = gpuArray(double(obj.vx));
    vy_gpu = gpuArray(double(obj.vy));
    sigmaxx_gpu = gpuArray(double(obj.sigmaxx));
    sigmayy_gpu = gpuArray(double(obj.sigmayy));
    sigmaxy_gpu = gpuArray(double(obj.sigmaxy));
    
    % PML相关数组
    memory_variables_gpu.dvx_dx = gpuArray(double(obj.memory_dvx_dx));
    memory_variables_gpu.dvy_dy = gpuArray(double(obj.memory_dvy_dy));
    memory_variables_gpu.dvy_dx = gpuArray(double(obj.memory_dvy_dx));
    memory_variables_gpu.dvx_dy = gpuArray(double(obj.memory_dvx_dy));
    memory_variables_gpu.dsigmaxx_dx = gpuArray(double(obj.memory_dsigmaxx_dx));
    memory_variables_gpu.dsigmaxy_dy = gpuArray(double(obj.memory_dsigmaxy_dy));
    memory_variables_gpu.dsigmaxy_dx = gpuArray(double(obj.memory_dsigmaxy_dx));
    memory_variables_gpu.dsigmayy_dy = gpuArray(double(obj.memory_dsigmayy_dy));
    
    % PML系数
    pml_coeffs_gpu.a_x = gpuArray(double(obj.a_x));
    pml_coeffs_gpu.a_x_half = gpuArray(double(obj.a_x_half));
    pml_coeffs_gpu.b_x = gpuArray(double(obj.b_x));
    pml_coeffs_gpu.b_x_half = gpuArray(double(obj.b_x_half));
    pml_coeffs_gpu.K_x = gpuArray(double(obj.K_x));
    pml_coeffs_gpu.K_x_half = gpuArray(double(obj.K_x_half));
    pml_coeffs_gpu.a_y = gpuArray(double(obj.a_y));
    pml_coeffs_gpu.a_y_half = gpuArray(double(obj.a_y_half));
    pml_coeffs_gpu.b_y = gpuArray(double(obj.b_y));
    pml_coeffs_gpu.b_y_half = gpuArray(double(obj.b_y_half));
    pml_coeffs_gpu.K_y = gpuArray(double(obj.K_y));
    pml_coeffs_gpu.K_y_half = gpuArray(double(obj.K_y_half));
end 