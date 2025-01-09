classdef VTI_WaveFieldSolver < handle
    properties
        % 网格参数
        NX              % x方向网格点数
        NY              % y方向网格点数
        DELTAX          % x方向网格间距
        DELTAY          % y方向网格间距
        
        % PML参数
        PML_XMIN
        PML_XMAX
        PML_YMIN
        PML_YMAX
        NPOINTS_PML
        
        % 材料参数
        c11             % 
        c13             % 
        c33             % 
        c44             % 
        rho             % 介质密度(kg/m³) 
        f0              % 震源频率(Hz) (保持标量)
        
        % 时间步进参数
        NSTEP           % 总时间步数
        DELTAT          % 时间步长
        
        % 震源参数
        ISOURCE         % 震源x位置索引 
        JSOURCE         % 震源y位置索引
        xsource         % 震源x坐标
        ysource         % 震源y坐标
        t0              % 时间延迟
        factor          % 震源强度因子
        ANGLE_FORCE     % 力的角度
        NSHOT           % 总炮数
        current_shot_number  % 当前炮号
        
        % 检波器参数
        NREC            % 检波器数量
        first_rec_x     % 第一个检波器x网格位置
        first_rec_y     % 第一个检波器y网格位置
        rec_x           % 检波器x网格位置数组
        rec_y           % 检波器y网格位置数组
        rec_dx          % 检波器x方向网格间隔
        rec_dy          % 检波器y方向网格间隔
        
        % 显示和常量参数
        IT_DISPLAY              % 显示间隔
        DEGREES_TO_RADIANS      % 角度转弧度系数
        ZERO                    % 零值常量
        HUGEVAL                 % 大数值常量
        STABILITY_THRESHOLD     % 稳定性阈值
        
        % PML基本参数
        NPOWER          % PML幂次
        K_MAX_PML       % 最大K值
        ALPHA_MAX_PML   % 最大alpha值
        
        % 波场变量
        vx              % x方向速度分量
        vy              % y方向速度分量
        sigmaxx         % xx方向应力分量
        sigmayy         % yy方向应力分量
        sigmaxy         % xy方向应力分量
        
        % PML记忆变量
        memory_dvx_dx       % vx对x的导数记忆变量
        memory_dvx_dy       % vx对y的导数记忆变量
        memory_dvy_dx       % vy对x的导数记忆变量
        memory_dvy_dy       % vy对y的导数记忆变量
        memory_dsigmaxx_dx  % sigmaxx对x的导数记忆变量
        memory_dsigmayy_dy  % sigmayy对y的导数记忆变量
        memory_dsigmaxy_dx  % sigmaxy对x的导数记忆变量
        memory_dsigmaxy_dy  % sigmaxy对y的导数记忆变量
        
        % 地震记录数组
        seismogram_vx       % x方向速度记录
        seismogram_vy       % y方向速度记录
        shot_records_vx     % x方向多炮记录
        shot_records_vy     % y方向多炮记录
        
        % 输出参数
        output_dir      % 输出目录
        save_snapshots  % 是否保存波场快照的控制开关
        
        % PML衰减参数 (x方向)
        d_x             % 衰减系数
        d_x_half        % 半网格点衰减系数
        K_x             % K系数
        K_x_half        % 半网格点K系数
        alpha_x         % alpha系数
        alpha_x_half    % 半网格点alpha系数
        a_x             % a系数
        a_x_half        % 半网格点a系数
        b_x             % b系数
        b_x_half        % 半网格点b系数
        
        % PML衰减参数 (y方向)
        d_y
        d_y_half
        K_y
        K_y_half
        alpha_y
        alpha_y_half
        a_y
        a_y_half
        b_y
        b_y_half
        
        % 计算模式
        compute_kernel    % 'cpu' 或 'cuda_mex'
        
       
%{
  % GPU变量
        vx_gpu
        vy_gpu
        sigmaxx_gpu
        sigmayy_gpu
        sigmaxy_gpu
        c11_gpu
        c13_gpu
        c33_gpu
        c44_gpu
        rho_gpu
        memory_dvx_dx_gpu
        memory_dvy_dy_gpu
        memory_dvy_dx_gpu
        memory_dvx_dy_gpu
        memory_dsigmaxx_dx_gpu
        memory_dsigmaxy_dy_gpu
        memory_dsigmaxy_dx_gpu
        memory_dsigmayy_dy_gpu
        K_x_gpu
        K_y_gpu
        K_x_half_gpu
        K_y_half_gpu
        a_x_gpu
        a_y_gpu
        a_x_half_gpu
        a_y_half_gpu
        b_x_gpu
        b_y_gpu
        b_x_half_gpu
        b_y_half_gpu 
%}

    end
    
    methods
        function obj = VTI_WaveFieldSolver(params)
            % 构造函数
            % params: 包含所有必要参数的结构体
            if nargin < 1
                error('必须提供参数结构体');
            end
            
            obj.initialize(params);
            obj.setup_receivers();
            
            % 设置输出目录
            current_dir = fileparts(fileparts(mfilename('fullpath')));
            obj.output_dir = fullfile(current_dir, 'data', 'output', 'wavefield', 'forward');
            
            if isfield(params, 'compute_kernel')
                obj.compute_kernel = params.compute_kernel;
            else
                obj.compute_kernel = 'cpu';
            end
        end
        
        function initialize(obj, params)
            % 初始化所有参数和数组
            obj.NX = params.NX;
            obj.NY = params.NY;
            obj.DELTAX = params.DELTAX;
            obj.DELTAY = params.DELTAY;
           
            obj.PML_XMIN = params.PML_XMIN;
            obj.PML_XMAX = params.PML_XMAX;
            obj.PML_YMIN = params.PML_YMIN;
            obj.PML_YMAX = params.PML_YMAX;
            obj.NPOINTS_PML = params.NPOINTS_PML;

            % 修改材料参数的初始化
            if isscalar(params.c11)
                % 如果输入是标量，扩展为整个模型大小的数组
                obj.c11 = ones(obj.NX, obj.NY) * params.c11;
                obj.c13 = ones(obj.NX, obj.NY) * params.c13;
                obj.c33 = ones(obj.NX, obj.NY) * params.c33;
                obj.c44 = ones(obj.NX, obj.NY) * params.c44;
                obj.rho = ones(obj.NX, obj.NY) * params.rho;
            else
                % 如果输入已经是数组，直接赋值
                obj.c11 = params.c11;
                obj.c13 = params.c13;
                obj.c33 = params.c33;
                obj.c44 = params.c44;
                obj.rho = params.rho;
                
                % 检查数组大小是否正确
                if ~all(size(obj.c11) == [obj.NX, obj.NY]) || ...
                   ~all(size(obj.c13) == [obj.NX, obj.NY]) || ...
                   ~all(size(obj.c33) == [obj.NX, obj.NY]) || ...
                   ~all(size(obj.c44) == [obj.NX, obj.NY]) || ...
                   ~all(size(obj.rho) == [obj.NX, obj.NY])
                    error('材料参数数组大小必须与模型大小(NX*NY)相匹配');
                end
            end
            
            obj.f0 = params.f0;  % 频率保持标量
            
            obj.NSTEP = params.NSTEP;
            obj.DELTAT = params.DELTAT;

            obj.t0 = 1.20/obj.f0;
            obj.factor = params.factor;  % 震源强度因子
            obj.ISOURCE = params.ISOURCE;
            obj.JSOURCE = params.JSOURCE;
            obj.xsource = (obj.ISOURCE - 1) * obj.DELTAX;
            obj.ysource = (obj.JSOURCE - 1) * obj.DELTAY;
            obj.ANGLE_FORCE = params.ANGLE_FORCE;  % 震源角度
            obj.NSHOT = params.NSHOT;  % 炮数
            obj.current_shot_number = 1;

            % 检波器参数
            obj.NREC = params.NREC;
            obj.first_rec_x = params.first_rec_x;
            obj.first_rec_y = params.first_rec_y;
            obj.rec_dx = params.rec_dx;
            obj.rec_dy = params.rec_dy; 

           % 显示和常量参数
            obj.IT_DISPLAY = params.IT_DISPLAY;
            obj.DEGREES_TO_RADIANS = pi / 180.0;
            obj.ZERO = 0.0;
            obj.HUGEVAL = 1.0e30;
            obj.STABILITY_THRESHOLD = 1.0e25;
           
            % PML基本参数
            obj.NPOWER = params.NPOWER;
            obj.K_MAX_PML = params.K_MAX_PML;
            obj.ALPHA_MAX_PML = 2.0 * pi * (obj.f0/2.0);
            
            % 初始化主要场变量数组
            obj.vx = zeros(obj.NX, obj.NY);      
            obj.vy = zeros(obj.NX, obj.NY);      
            obj.sigmaxx = zeros(obj.NX, obj.NY); 
            obj.sigmayy = zeros(obj.NX, obj.NY); 
            obj.sigmaxy = zeros(obj.NX, obj.NY); 
            
            % 速度导数的记忆变量
            obj.memory_dvx_dx = zeros(obj.NX, obj.NY);
            obj.memory_dvx_dy = zeros(obj.NX, obj.NY);
            obj.memory_dvy_dx = zeros(obj.NX, obj.NY);
            obj.memory_dvy_dy = zeros(obj.NX, obj.NY);
            
            % 应力导数的记忆变量
            obj.memory_dsigmaxx_dx = zeros(obj.NX, obj.NY);
            obj.memory_dsigmayy_dy = zeros(obj.NX, obj.NY);
            obj.memory_dsigmaxy_dx = zeros(obj.NX, obj.NY);
            obj.memory_dsigmaxy_dy = zeros(obj.NX, obj.NY);
        
            % 初始化地震记录数组
            obj.seismogram_vx = zeros(obj.NSTEP, obj.NREC);
            obj.seismogram_vy = zeros(obj.NSTEP, obj.NREC);
            obj.shot_records_vx = zeros(obj.NSHOT, obj.NSTEP, obj.NREC);
            obj.shot_records_vy = zeros(obj.NSHOT, obj.NSTEP, obj.NREC);
            
            % x方向的衰减参数
            obj.d_x = zeros(obj.NX, 1);
            obj.d_x_half = zeros(obj.NX, 1);
            obj.K_x = ones(obj.NX, 1);
            obj.K_x_half = ones(obj.NX, 1);
            obj.alpha_x = zeros(obj.NX, 1);
            obj.alpha_x_half = zeros(obj.NX, 1);
            obj.a_x = zeros(obj.NX, 1);
            obj.a_x_half = zeros(obj.NX, 1);
            obj.b_x = zeros(obj.NX, 1);
            obj.b_x_half = zeros(obj.NX, 1);
            
            % y方向的衰减参数
            obj.d_y = zeros(obj.NY, 1);
            obj.d_y_half = zeros(obj.NY, 1);
            obj.K_y = ones(obj.NY, 1);
            obj.K_y_half = ones(obj.NY, 1);
            obj.alpha_y = zeros(obj.NY, 1);
            obj.alpha_y_half = zeros(obj.NY, 1);
            obj.a_y = zeros(obj.NY, 1);
            obj.a_y_half = zeros(obj.NY, 1);
            obj.b_y = zeros(obj.NY, 1);
            obj.b_y_half = zeros(obj.NY, 1);
            
            % 波场快照保存控制
            if isfield(params, 'save_snapshots')
                obj.save_snapshots = params.save_snapshots;
            else
                obj.save_snapshots = false;
            end
        end
        
        
%{
        function initialize_gpu_arrays(obj)
                    % 场变量
                    obj.vx_gpu = gpuArray(obj.vx);
                    obj.vy_gpu = gpuArray(obj.vy);
                    obj.sigmaxx_gpu = gpuArray(obj.sigmaxx);
                    obj.sigmayy_gpu = gpuArray(obj.sigmayy);
                    obj.sigmaxy_gpu = gpuArray(obj.sigmaxy);
                    
                    % 材料参数
                    obj.c11_gpu = gpuArray(obj.c11);
                    obj.c13_gpu = gpuArray(obj.c13);
                    obj.c33_gpu = gpuArray(obj.c33);
                    obj.c44_gpu = gpuArray(obj.c44);
                    obj.rho_gpu = gpuArray(obj.rho);
                    
                    % PML记忆变量
                    obj.memory_dvx_dx_gpu = gpuArray(obj.memory_dvx_dx);
                    obj.memory_dvy_dy_gpu = gpuArray(obj.memory_dvy_dy);
                    obj.memory_dvy_dx_gpu = gpuArray(obj.memory_dvy_dx);
                    obj.memory_dvx_dy_gpu = gpuArray(obj.memory_dvx_dy);
                    obj.memory_dsigmaxx_dx_gpu = gpuArray(obj.memory_dsigmaxx_dx);
                    obj.memory_dsigmaxy_dy_gpu = gpuArray(obj.memory_dsigmaxy_dy);
                    obj.memory_dsigmaxy_dx_gpu = gpuArray(obj.memory_dsigmaxy_dx);
                    obj.memory_dsigmayy_dy_gpu = gpuArray(obj.memory_dsigmayy_dy);
                    
                    % PML系数
                    obj.K_x_gpu = gpuArray(obj.K_x);
                    obj.K_y_gpu = gpuArray(obj.K_y);
                    obj.K_x_half_gpu = gpuArray(obj.K_x_half);
                    obj.K_y_half_gpu = gpuArray(obj.K_y_half);
                    obj.a_x_gpu = gpuArray(obj.a_x);
                    obj.a_y_gpu = gpuArray(obj.a_y);
                    obj.a_x_half_gpu = gpuArray(obj.a_x_half);
                    obj.a_y_half_gpu = gpuArray(obj.a_y_half);
                    obj.b_x_gpu = gpuArray(obj.b_x);
                    obj.b_y_gpu = gpuArray(obj.b_y);
                    obj.b_x_half_gpu = gpuArray(obj.b_x_half);
                    obj.b_y_half_gpu = gpuArray(obj.b_y_half);
                end 
%}


        function add_source(obj, it)
            % 添加震源（在指定网格点处添加力矢量）
            a = pi * pi * obj.f0 * obj.f0;
            t = (it-1) * obj.DELTAT;
            source_term = -obj.factor * 2.0 * a * (t - obj.t0) * exp(-a * (t - obj.t0) * (t - obj.t0));
            
            % 根据震源角度分解力矢量到x和y方向
            force_x = sin(obj.ANGLE_FORCE * obj.DEGREES_TO_RADIANS) * source_term;
            force_y = cos(obj.ANGLE_FORCE * obj.DEGREES_TO_RADIANS) * source_term;
            
            % 获取震源位置
            i = obj.ISOURCE;  % 震源x坐标
            j = obj.JSOURCE;  % 震源y坐标
            
            % 将力添加到速度场中，使用震源位置处的密度值
            obj.vx(i,j) = obj.vx(i,j) + force_x * obj.DELTAT / obj.rho(i,j);  % 更新x方向速度
            obj.vy(i,j) = obj.vy(i,j) + force_y * obj.DELTAT / obj.rho(i,j);  % 更新y方向速度
        end
        
        function setup_receivers(obj)
            % 初始化检波器位置数组（网格点位置）
            obj.rec_x = zeros(obj.NREC, 1);
            obj.rec_y = zeros(obj.NREC, 1);
            
            % 计算每个检波器的网格位置
            for i = 1:obj.NREC
                obj.rec_x(i) = obj.first_rec_x + (i-1) * obj.rec_dx;
                obj.rec_y(i) = obj.first_rec_y + (i-1) * obj.rec_dy;
            end
            
            % 检查网格位置是否在有效范围内
            if any(obj.rec_x < 1 | obj.rec_x > obj.NX | ...
                   obj.rec_y < 1 | obj.rec_y > obj.NY)
                error('检波器位置超出计算网格范围');
            end
        end
        
        function setup_pml_boundary(obj)
            % 检查稳定性 - 使用数组运算并检查所有网格点
            if any((obj.c11(:).*obj.c33(:) - obj.c13(:).^2) <= 0)
                error('VTI材料定义错误：c11*c33 - c13^2 必须在所有网格点上大于0');
            end

            % 检查各向异性材料PML模型的稳定性条件
            % 计算整个模型空间的稳定性判据，并取最大值
            aniso_stability_criterion = max(max(((obj.c13 + obj.c44).^2 - obj.c11.*(obj.c33-obj.c44)) .* ...
                                              ((obj.c13 + obj.c44).^2 + obj.c44.*(obj.c33-obj.c44))));
            % fprintf('Becache等人2003年提出的PML各向异性稳定性判据最大值 = %f\n', aniso_stability_criterion);
            if aniso_stability_criterion > 0.0 && (obj.PML_XMIN || obj.PML_XMAX || ...
                                                  obj.PML_YMIN || obj.PML_YMAX)
                warning('警告：对于该各向异性材料，PML模型在条件1下在某些位置数学上本质上不稳定');
            end
            
            % 检查第二个稳定性条件
            aniso2 = max(max((obj.c13 + 2*obj.c44).^2 - obj.c11.*obj.c33));
            % fprintf('Becache等人2003年提出的PML aniso2稳定性判据最大值 = %f\n', aniso2);
            if aniso2 > 0.0 && (obj.PML_XMIN || obj.PML_XMAX || ...
                                obj.PML_YMIN || obj.PML_YMAX)
                warning('警告：对于该各向异性材料，PML模型在条件2下在某些位置数学上本质上不稳定');
            end
            
            % 检查第三个稳定性条件
            aniso3 = max(max((obj.c13 + obj.c44).^2 - obj.c11.*obj.c33 - obj.c44.^2));
            % fprintf('Becache等人2003年提出的PML aniso3稳定性判据最大值 = %f\n', aniso3);
            if aniso3 > 0.0 && (obj.PML_XMIN || obj.PML_XMAX || ...
                                obj.PML_YMIN || obj.PML_YMAX)
                warning('警告：对于该各向异性材料，PML模型在条件3下在某些位置数学上本质上不稳定');
            end
        end
        
        function setup_pml_boundary_x(obj)
            % 计算准P波最大速度，使用数组最大值
            quasi_cp_max = max(max(max(sqrt(obj.c33./obj.rho)), max(sqrt(obj.c11./obj.rho))));
            
            % 定义PML区域的吸收层厚度
            thickness_PML_x = obj.NPOINTS_PML * obj.DELTAX;
            
            % 设置反射系数
            Rcoef = 0.001;
            
            % 计算衰减系数d0
            d0_x = -(obj.NPOWER + 1) * quasi_cp_max * log(Rcoef) / (2.0 * thickness_PML_x);
            fprintf('d0_x = %f\n', d0_x);
            
            % 设置边界位置
            xoriginleft = thickness_PML_x;
            xoriginright = (obj.NX-1)*obj.DELTAX - thickness_PML_x;
            
            % 创建完整的网格坐标数组
            x_vals = (0:obj.NX-1)' * obj.DELTAX;
            x_vals_half = x_vals + obj.DELTAX/2.0;
            
            % 预分配完整大小的数组 - 整数网格点
            d_x_left = zeros(obj.NX, 1);
            d_x_right = zeros(obj.NX, 1);
            K_x_left = ones(obj.NX, 1);
            K_x_right = ones(obj.NX, 1);
            alpha_x_left = zeros(obj.NX, 1);
            alpha_x_right = zeros(obj.NX, 1);
            
            % 预分配完整大小的数组 - 半网格点
            d_x_half_left = zeros(obj.NX, 1);
            d_x_half_right = zeros(obj.NX, 1);
            K_x_half_left = ones(obj.NX, 1);
            K_x_half_right = ones(obj.NX, 1);
            alpha_x_half_left = zeros(obj.NX, 1);
            alpha_x_half_right = zeros(obj.NX, 1);
            
            % 左边界PML计算
            if obj.PML_XMIN
                % 整数网格点
                abscissa_in_PML = xoriginleft - x_vals;
                mask = abscissa_in_PML >= 0.0;
                abscissa_normalized = zeros(size(x_vals));
                abscissa_normalized(mask) = abscissa_in_PML(mask) / thickness_PML_x;
                
                d_x_left(mask) = d0_x * abscissa_normalized(mask).^obj.NPOWER;
                K_x_left(mask) = 1.0 + (obj.K_MAX_PML - 1.0) * abscissa_normalized(mask).^obj.NPOWER;
                alpha_x_left(mask) = obj.ALPHA_MAX_PML * (1.0 - abscissa_normalized(mask));
                
                % 半网格点
                abscissa_in_PML_half = xoriginleft - x_vals_half;
                mask_half = abscissa_in_PML_half >= 0.0;
                abscissa_normalized_half = zeros(size(x_vals_half));
                abscissa_normalized_half(mask_half) = abscissa_in_PML_half(mask_half) / thickness_PML_x;
                
                d_x_half_left(mask_half) = d0_x * abscissa_normalized_half(mask_half).^obj.NPOWER;
                K_x_half_left(mask_half) = 1.0 + (obj.K_MAX_PML - 1.0) * abscissa_normalized_half(mask_half).^obj.NPOWER;
                alpha_x_half_left(mask_half) = obj.ALPHA_MAX_PML * (1.0 - abscissa_normalized_half(mask_half));
            end
            
            % 右边界PML计算
            if obj.PML_XMAX
                % 整数网格点
                abscissa_in_PML = x_vals - xoriginright;
                mask = abscissa_in_PML >= 0.0;
                abscissa_normalized = zeros(size(x_vals));
                abscissa_normalized(mask) = abscissa_in_PML(mask) / thickness_PML_x;
                
                d_x_right(mask) = d0_x * abscissa_normalized(mask).^obj.NPOWER;
                K_x_right(mask) = 1.0 + (obj.K_MAX_PML - 1.0) * abscissa_normalized(mask).^obj.NPOWER;
                alpha_x_right(mask) = obj.ALPHA_MAX_PML * (1.0 - abscissa_normalized(mask));
                
                % 半网格点
                abscissa_in_PML_half = x_vals_half - xoriginright;
                mask_half = abscissa_in_PML_half >= 0.0;
                abscissa_normalized_half = zeros(size(x_vals_half));
                abscissa_normalized_half(mask_half) = abscissa_in_PML_half(mask_half) / thickness_PML_x;
                
                d_x_half_right(mask_half) = d0_x * abscissa_normalized_half(mask_half).^obj.NPOWER;
                K_x_half_right(mask_half) = 1.0 + (obj.K_MAX_PML - 1.0) * abscissa_normalized_half(mask_half).^obj.NPOWER;
                alpha_x_half_right(mask_half) = obj.ALPHA_MAX_PML * (1.0 - abscissa_normalized_half(mask_half));
            end
            
            % 合并左右边界的结果 - 整数网格点
            obj.d_x = d_x_left + d_x_right;
            obj.K_x = max(K_x_left, K_x_right);
            obj.alpha_x = alpha_x_left + alpha_x_right;
            
            % 合并左右边界的结果 - 半网格点
            obj.d_x_half = d_x_half_left + d_x_half_right;
            obj.K_x_half = max(K_x_half_left, K_x_half_right);
            obj.alpha_x_half = alpha_x_half_left + alpha_x_half_right;
            
            % 确保alpha值非负
            obj.alpha_x = max(obj.alpha_x, obj.ZERO);
            obj.alpha_x_half = max(obj.alpha_x_half, obj.ZERO);
            
            % 计算b系数
            obj.b_x = exp(-(obj.d_x ./ obj.K_x + obj.alpha_x) * obj.DELTAT);
            obj.b_x_half = exp(-(obj.d_x_half ./ obj.K_x_half + obj.alpha_x_half) * obj.DELTAT);
            
            % 计算a系数 - 整数网格点
            obj.a_x = zeros(size(obj.d_x));
            mask = abs(obj.d_x) > 1.0e-6;
            obj.a_x(mask) = obj.d_x(mask) .* (obj.b_x(mask) - 1.0) ./ ...
                (obj.K_x(mask) .* (obj.d_x(mask) + obj.K_x(mask) .* obj.alpha_x(mask)));
            
            % 计算a系数 - 半网格点
            obj.a_x_half = zeros(size(obj.d_x_half));
            mask_half = abs(obj.d_x_half) > 1.0e-6;
            obj.a_x_half(mask_half) = obj.d_x_half(mask_half) .* (obj.b_x_half(mask_half) - 1.0) ./ ...
                (obj.K_x_half(mask_half) .* (obj.d_x_half(mask_half) + obj.K_x_half(mask_half) .* obj.alpha_x_half(mask_half)));
            
            % 打印数组参数统计信息
            % fprintf('\n=== X方向CPML数组参数统计 ===\n');
            % fprintf('d_x: min=%f, max=%f, mean=%f\n', min(obj.d_x), max(obj.d_x), mean(obj.d_x));
            % fprintf('d_x_half: min=%f, max=%f, mean=%f\n', min(obj.d_x_half), max(obj.d_x_half), mean(obj.d_x_half));
            % fprintf('K_x: min=%f, max=%f, mean=%f\n', min(obj.K_x), max(obj.K_x), mean(obj.K_x));
            % fprintf('K_x_half: min=%f, max=%f, mean=%f\n', min(obj.K_x_half), max(obj.K_x_half), mean(obj.K_x_half));
            % fprintf('alpha_x: min=%f, max=%f, mean=%f\n', min(obj.alpha_x), max(obj.alpha_x), mean(obj.alpha_x));
            % fprintf('alpha_x_half: min=%f, max=%f, mean=%f\n', min(obj.alpha_x_half), max(obj.alpha_x_half), mean(obj.alpha_x_half));
            % fprintf('b_x: min=%f, max=%f, mean=%f\n', min(obj.b_x), max(obj.b_x), mean(obj.b_x));
            % fprintf('b_x_half: min=%f, max=%f, mean=%f\n', min(obj.b_x_half), max(obj.b_x_half), mean(obj.b_x_half));
            % fprintf('a_x: min=%f, max=%f, mean=%f\n', min(obj.a_x), max(obj.a_x), mean(obj.a_x));
            % fprintf('a_x_half: min=%f, max=%f, mean=%f\n', min(obj.a_x_half), max(obj.a_x_half), mean(obj.a_x_half));
        end
        
        function setup_pml_boundary_y(obj)
            % 计算准P波最大速度，使用数组最大值
            quasi_cp_max = max(max(max(sqrt(obj.c33./obj.rho)), max(sqrt(obj.c11./obj.rho))));
            
            % 定义PML区域的吸收层厚度
            thickness_PML_y = obj.NPOINTS_PML * obj.DELTAY;
            
            % 设置反射系数
            Rcoef = 0.001;
            
            % 计算衰减系数d0
            d0_y = -(obj.NPOWER + 1) * quasi_cp_max * log(Rcoef) / (2.0 * thickness_PML_y);
            fprintf('d0_y = %f\n', d0_y);
            
            % 设置边界位置
            yoriginleft = thickness_PML_y;
            yoriginright = (obj.NY-1)*obj.DELTAY - thickness_PML_y;
            
            % 创建完整的网格坐标数组
            y_vals = (0:obj.NY-1)' * obj.DELTAY;
            y_vals_half = y_vals + obj.DELTAY/2.0;
            
            % 预分配完整大小的数组 - 整数网格点
            d_y_left = zeros(obj.NY, 1);
            d_y_right = zeros(obj.NY, 1);
            K_y_left = ones(obj.NY, 1);
            K_y_right = ones(obj.NY, 1);
            alpha_y_left = zeros(obj.NY, 1);
            alpha_y_right = zeros(obj.NY, 1);
            
            % 预分配完整大小的数组 - 半网格点
            d_y_half_left = zeros(obj.NY, 1);
            d_y_half_right = zeros(obj.NY, 1);
            K_y_half_left = ones(obj.NY, 1);
            K_y_half_right = ones(obj.NY, 1);
            alpha_y_half_left = zeros(obj.NY, 1);
            alpha_y_half_right = zeros(obj.NY, 1);
            
            % 下边界PML计算
            if obj.PML_YMIN
                % 整数网格点
                abscissa_in_PML = yoriginleft - y_vals;
                mask = abscissa_in_PML >= 0.0;
                abscissa_normalized = zeros(size(y_vals));
                abscissa_normalized(mask) = abscissa_in_PML(mask) / thickness_PML_y;
                
                d_y_left(mask) = d0_y * abscissa_normalized(mask).^obj.NPOWER;
                K_y_left(mask) = 1.0 + (obj.K_MAX_PML - 1.0) * abscissa_normalized(mask).^obj.NPOWER;
                alpha_y_left(mask) = obj.ALPHA_MAX_PML * (1.0 - abscissa_normalized(mask));
                
                % 半网格点
                abscissa_in_PML_half = yoriginleft - y_vals_half;
                mask_half = abscissa_in_PML_half >= 0.0;
                abscissa_normalized_half = zeros(size(y_vals_half));
                abscissa_normalized_half(mask_half) = abscissa_in_PML_half(mask_half) / thickness_PML_y;
                
                d_y_half_left(mask_half) = d0_y * abscissa_normalized_half(mask_half).^obj.NPOWER;
                K_y_half_left(mask_half) = 1.0 + (obj.K_MAX_PML - 1.0) * abscissa_normalized_half(mask_half).^obj.NPOWER;
                alpha_y_half_left(mask_half) = obj.ALPHA_MAX_PML * (1.0 - abscissa_normalized_half(mask_half));
            end
            
            % 上边界PML计算
            if obj.PML_YMAX
                % 整数网格点
                abscissa_in_PML = y_vals - yoriginright;
                mask = abscissa_in_PML >= 0.0;
                abscissa_normalized = zeros(size(y_vals));
                abscissa_normalized(mask) = abscissa_in_PML(mask) / thickness_PML_y;
                
                d_y_right(mask) = d0_y * abscissa_normalized(mask).^obj.NPOWER;
                K_y_right(mask) = 1.0 + (obj.K_MAX_PML - 1.0) * abscissa_normalized(mask).^obj.NPOWER;
                alpha_y_right(mask) = obj.ALPHA_MAX_PML * (1.0 - abscissa_normalized(mask));
                
                % 半网格点
                abscissa_in_PML_half = y_vals_half - yoriginright;
                mask_half = abscissa_in_PML_half >= 0.0;
                abscissa_normalized_half = zeros(size(y_vals_half));
                abscissa_normalized_half(mask_half) = abscissa_in_PML_half(mask_half) / thickness_PML_y;
                
                d_y_half_right(mask_half) = d0_y * abscissa_normalized_half(mask_half).^obj.NPOWER;
                K_y_half_right(mask_half) = 1.0 + (obj.K_MAX_PML - 1.0) * abscissa_normalized_half(mask_half).^obj.NPOWER;
                alpha_y_half_right(mask_half) = obj.ALPHA_MAX_PML * (1.0 - abscissa_normalized_half(mask_half));
            end
            
            % 合并上下边界的结果 - 整数网格点
            obj.d_y = d_y_left + d_y_right;
            obj.K_y = max(K_y_left, K_y_right);
            obj.alpha_y = alpha_y_left + alpha_y_right;
            
            % 合并上下边界的结果 - 半网格点
            obj.d_y_half = d_y_half_left + d_y_half_right;
            obj.K_y_half = max(K_y_half_left, K_y_half_right);
            obj.alpha_y_half = alpha_y_half_left + alpha_y_half_right;
            
            % 确保alpha值非负
            obj.alpha_y = max(obj.alpha_y, obj.ZERO);
            obj.alpha_y_half = max(obj.alpha_y_half, obj.ZERO);
            
            % 计算b系数
            obj.b_y = exp(-(obj.d_y ./ obj.K_y + obj.alpha_y) * obj.DELTAT);
            obj.b_y_half = exp(-(obj.d_y_half ./ obj.K_y_half + obj.alpha_y_half) * obj.DELTAT);
            
            % 计算a系数 - 整数网格点
            obj.a_y = zeros(size(obj.d_y));
            mask = abs(obj.d_y) > 1.0e-6;
            obj.a_y(mask) = obj.d_y(mask) .* (obj.b_y(mask) - 1.0) ./ ...
                (obj.K_y(mask) .* (obj.d_y(mask) + obj.K_y(mask) .* obj.alpha_y(mask)));
            
            % 计算a系数 - 半网格点
            obj.a_y_half = zeros(size(obj.d_y_half));
            mask_half = abs(obj.d_y_half) > 1.0e-6;
            obj.a_y_half(mask_half) = obj.d_y_half(mask_half) .* (obj.b_y_half(mask_half) - 1.0) ./ ...
                (obj.K_y_half(mask_half) .* (obj.d_y_half(mask_half) + obj.K_y_half(mask_half) .* obj.alpha_y_half(mask_half)));
            
            % 打印数组参数统计信息
            % fprintf('\n=== Y方向CPML数组参数统计 ===\n');
            % fprintf('d_y: min=%f, max=%f, mean=%f\n', min(obj.d_y), max(obj.d_y), mean(obj.d_y));
            % fprintf('d_y_half: min=%f, max=%f, mean=%f\n', min(obj.d_y_half), max(obj.d_y_half), mean(obj.d_y_half));
            % fprintf('K_y: min=%f, max=%f, mean=%f\n', min(obj.K_y), max(obj.K_y), mean(obj.K_y));
            % fprintf('K_y_half: min=%f, max=%f, mean=%f\n', min(obj.K_y_half), max(obj.K_y_half), mean(obj.K_y_half));
            % fprintf('alpha_y: min=%f, max=%f, mean=%f\n', min(obj.alpha_y), max(obj.alpha_y), mean(obj.alpha_y));
            % fprintf('alpha_y_half: min=%f, max=%f, mean=%f\n', min(obj.alpha_y_half), max(obj.alpha_y_half), mean(obj.alpha_y_half));
            % fprintf('b_y: min=%f, max=%f, mean=%f\n', min(obj.b_y), max(obj.b_y), mean(obj.b_y));
            % fprintf('b_y_half: min=%f, max=%f, mean=%f\n', min(obj.b_y_half), max(obj.b_y_half), mean(obj.b_y_half));
            % fprintf('a_y: min=%f, max=%f, mean=%f\n', min(obj.a_y), max(obj.a_y), mean(obj.a_y));
            % fprintf('a_y_half: min=%f, max=%f, mean=%f\n', min(obj.a_y_half), max(obj.a_y_half), mean(obj.a_y_half));
        end
        
        function compute_wave_propagation(obj)
            switch obj.compute_kernel
                case 'cpu'
                    obj.compute_wave_propagation_cpu();  % 原来的CPU实现
                case 'cuda_mex'
                    % 将来添加的CUDA MEX实现
                    error('CUDA MEX kernel not implemented yet');
                otherwise
                    error('Unknown compute kernel type');
            end
        end

        function compute_wave_propagation_cpu(obj)

            % 使用类成员变量
            dx = obj.DELTAX;
            dy = obj.DELTAY;
            dt = obj.DELTAT;
            
            % 计算应力场 sigmaxx 和 sigmayy
            for j = 2:obj.NY
                for i = 1:obj.NX-1
                    % 计算速度梯度
                    value_dvx_dx = (obj.vx(i+1,j) - obj.vx(i,j)) / dx;
                    value_dvy_dy = (obj.vy(i,j) - obj.vy(i,j-1)) / dy;
                    
                    % 更新PML记忆变量
                    obj.memory_dvx_dx(i,j) = obj.b_x_half(i) * obj.memory_dvx_dx(i,j) + ...
                                            obj.a_x_half(i) * value_dvx_dx;
                    obj.memory_dvy_dy(i,j) = obj.b_y(j) * obj.memory_dvy_dy(i,j) + ...
                                            obj.a_y(j) * value_dvy_dy;
                    
                    % 计算最终值
                    value_dvx_dx = value_dvx_dx / obj.K_x_half(i) + obj.memory_dvx_dx(i,j);
                    value_dvy_dy = value_dvy_dy / obj.K_y(j) + obj.memory_dvy_dy(i,j);
                    
                    % 在这里，cij是二维数组，(i,j)索引确保维度匹配
                    % 每个点的应力更新都使用对应位置的材料参数
                    obj.sigmaxx(i,j) = obj.sigmaxx(i,j) + dt * (...
                        obj.c11(i,j) * value_dvx_dx + ...  % c11在点(i,j)的值
                        obj.c13(i,j) * value_dvy_dy);      % c13在点(i,j)的值
                    
                    obj.sigmayy(i,j) = obj.sigmayy(i,j) + dt * (...
                        obj.c13(i,j) * value_dvx_dx + ...  % c13在点(i,j)的值
                        obj.c33(i,j) * value_dvy_dy);      % c33在点(i,j)的值
                end
            end
            
            % 计算剪应力 sigmaxy
            for j = 1:obj.NY-1
                for i = 2:obj.NX
                    value_dvy_dx = (obj.vy(i,j) - obj.vy(i-1,j)) / dx;
                    value_dvx_dy = (obj.vx(i,j+1) - obj.vx(i,j)) / dy;
                    
                    obj.memory_dvy_dx(i,j) = obj.b_x(i) * obj.memory_dvy_dx(i,j) + ...
                                            obj.a_x(i) * value_dvy_dx;
                    obj.memory_dvx_dy(i,j) = obj.b_y_half(j) * obj.memory_dvx_dy(i,j) + ...
                                            obj.a_y_half(j) * value_dvx_dy;
                    
                    value_dvy_dx = value_dvy_dx / obj.K_x(i) + obj.memory_dvy_dx(i,j);
                    value_dvx_dy = value_dvx_dy / obj.K_y_half(j) + obj.memory_dvx_dy(i,j);
                    
                    % c44也是二维数组，使用对应点的值
                    obj.sigmaxy(i,j) = obj.sigmaxy(i,j) + ...
                        obj.c44(i,j) * (value_dvy_dx + value_dvx_dy) * dt;
                end
            end
            
            % 计算x方向速度场
            for j = 2:obj.NY
                for i = 2:obj.NX
                    value_dsigmaxx_dx = (obj.sigmaxx(i,j) - obj.sigmaxx(i-1,j)) / dx;
                    value_dsigmaxy_dy = (obj.sigmaxy(i,j) - obj.sigmaxy(i,j-1)) / dy;
                    
                    obj.memory_dsigmaxx_dx(i,j) = obj.b_x(i) * obj.memory_dsigmaxx_dx(i,j) + ...
                                                 obj.a_x(i) * value_dsigmaxx_dx;
                    obj.memory_dsigmaxy_dy(i,j) = obj.b_y(j) * obj.memory_dsigmaxy_dy(i,j) + ...
                                                 obj.a_y(j) * value_dsigmaxy_dy;
                    
                    value_dsigmaxx_dx = value_dsigmaxx_dx / obj.K_x(i) + obj.memory_dsigmaxx_dx(i,j);
                    value_dsigmaxy_dy = value_dsigmaxy_dy / obj.K_y(j) + obj.memory_dsigmaxy_dy(i,j);
                    
                    % rho也是二维数组
                    obj.vx(i,j) = obj.vx(i,j) + ...
                        (value_dsigmaxx_dx + value_dsigmaxy_dy) * dt / obj.rho(i,j);
                end
            end
            
            % 计算y方向速度场
            for j = 1:obj.NY-1
                for i = 1:obj.NX-1
                    value_dsigmaxy_dx = (obj.sigmaxy(i+1,j) - obj.sigmaxy(i,j)) / dx;
                    value_dsigmayy_dy = (obj.sigmayy(i,j+1) - obj.sigmayy(i,j)) / dy;
                    
                    obj.memory_dsigmaxy_dx(i,j) = obj.b_x_half(i) * obj.memory_dsigmaxy_dx(i,j) + ...
                                                 obj.a_x_half(i) * value_dsigmaxy_dx;
                    obj.memory_dsigmayy_dy(i,j) = obj.b_y_half(j) * obj.memory_dsigmayy_dy(i,j) + ...
                                                 obj.a_y_half(j) * value_dsigmayy_dy;
                    
                    value_dsigmaxy_dx = value_dsigmaxy_dx / obj.K_x_half(i) + obj.memory_dsigmaxy_dx(i,j);
                    value_dsigmayy_dy = value_dsigmayy_dy / obj.K_y_half(j) + obj.memory_dsigmayy_dy(i,j);
                    
                    % rho是二维数组
                    obj.vy(i,j) = obj.vy(i,j) + ...
                        (value_dsigmaxy_dx + value_dsigmayy_dy) * dt / obj.rho(i,j);
                end
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%可以删除了，切片不对。%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
        %{
        function compute_wave_propagation_gpu(obj)
                    t_start = tic;
                    fprintf('开始GPU波场计算...\n');
                    
                    % 基本参数
                    dx = obj.DELTAX;
                    dy = obj.DELTAY;
                    dt = obj.DELTAT;
                    
                    % 计算应力场 sigmaxx 和 sigmayy
                    t1 = tic;
                    fprintf('计算应力场...\n');
                    
                    % 使用数组操作计算速度梯度
                    dvx_dx = diff(obj.vx_gpu, 1, 1) / dx;  % 在x方向计算差分 [NX-1, NY]
                    dvy_dy = diff(obj.vy_gpu, 1, 2) / dy;  % 在y方向计算差分 [NX, NY-1]
                    
                    % 现在可以打印维度信息了
                    fprintf('数组维度信息:\n');
                    fprintf('dvx_dx: [%d, %d]\n', size(dvx_dx,1), size(dvx_dx,2));
                    fprintf('dvy_dy: [%d, %d]\n', size(dvy_dy,1), size(dvy_dy,2));
                    fprintf('c11_gpu: [%d, %d]\n', size(obj.c11_gpu,1), size(obj.c11_gpu,2));
                    
                    % 更新PML记忆变量（使用数组广播）
                    obj.memory_dvx_dx_gpu(1:end-1,:) = obj.b_x_half_gpu(1:end-1) .* obj.memory_dvx_dx_gpu(1:end-1,:) + ...
                                                obj.a_x_half_gpu(1:end-1) .* dvx_dx;
                    obj.memory_dvy_dy_gpu(:,1:end-1) = obj.b_y_gpu(1:end-1)' .* obj.memory_dvy_dy_gpu(:,1:end-1) + ...
                                                obj.a_y_gpu(1:end-1)' .* dvy_dy;
                    
                    % 计算最终值（确保维度匹配）
                    dvx_dx_final = dvx_dx ./ obj.K_x_half_gpu(1:end-1) + obj.memory_dvx_dx_gpu(1:end-1,:);
                    dvy_dy_final = dvy_dy ./ obj.K_y_gpu(1:end-1)' + obj.memory_dvy_dy_gpu(:,1:end-1);
                    
                    % 更新应力场（确保所有数组维度匹配）
                    % 注意：我们需要确保所有参与计算的数组具有相同的维度
                    obj.sigmaxx_gpu(1:end-1,1:end-1) = obj.sigmaxx_gpu(1:end-1,1:end-1) + dt * (...
                        obj.c11_gpu(1:end-1,1:end-1) .* dvx_dx_final(:,1:end-1) + ...
                        obj.c13_gpu(1:end-1,1:end-1) .* dvy_dy_final(1:end-1,:));
                    
                    obj.sigmayy_gpu(1:end-1,1:end-1) = obj.sigmayy_gpu(1:end-1,1:end-1) + dt * (...
                        obj.c13_gpu(1:end-1,1:end-1) .* dvx_dx_final(:,1:end-1) + ...
                        obj.c33_gpu(1:end-1,1:end-1) .* dvy_dy_final(1:end-1,:));
                    
                    fprintf('应力场计算完成，用时: %.3f秒\n', toc(t1));
                    
                    % 计算剪应力 sigmaxy
                    t2 = tic;
                    fprintf('计算剪应力...\n');
                    
                    % 使用数组操作计算速度梯度
                    dvy_dx = diff(obj.vy_gpu, 1, 1) / dx;  % [NX-1, NY]
                    dvx_dy = diff(obj.vx_gpu, 1, 2) / dy;  % [NX, NY-1]
                    
                    % 打印所有相关数组的维度
                    fprintf('数组维度检查:\n');
                    fprintf('dvy_dx: [%d, %d]\n', size(dvy_dx,1), size(dvy_dx,2));
                    fprintf('dvx_dy: [%d, %d]\n', size(dvx_dy,1), size(dvx_dy,2));
                    fprintf('b_x_gpu: [%d, 1]\n', size(obj.b_x_gpu,1));
                    fprintf('memory_dvy_dx_gpu: [%d, %d]\n', size(obj.memory_dvy_dx_gpu,1), size(obj.memory_dvy_dx_gpu,2));
                    
                    % 修正索引范围，确保维度匹配
                    % 注意：我们需要确保所有切片产生相同大小的数组
                    memory_slice = obj.memory_dvy_dx_gpu(2:end-1,1:end-1);
                    b_x_slice = obj.b_x_gpu(2:end-1);
                    dvy_dx_slice = dvy_dx(1:end-1,1:end-1);  % 修改这里，确保与memory_slice维度匹配
                    
                    fprintf('切片后维度检查:\n');
                    fprintf('memory_slice: [%d, %d]\n', size(memory_slice,1), size(memory_slice,2));
                    fprintf('b_x_slice: [%d, 1]\n', size(b_x_slice,1));
                    fprintf('dvy_dx_slice: [%d, %d]\n', size(dvy_dx_slice,1), size(dvy_dx_slice,2));
                    
                    % 更新memory_dvy_dx_gpu
                    obj.memory_dvy_dx_gpu(2:end-1,1:end-1) = b_x_slice .* memory_slice + ...
                                                        obj.a_x_gpu(2:end-1) .* dvy_dx_slice;
                    
                    % 更新memory_dvx_dy_gpu
                    memory_dvx_slice = obj.memory_dvx_dy_gpu(2:end-1,1:end-1);
                    b_y_slice = obj.b_y_half_gpu(1:end-1);
                    dvx_dy_slice = dvx_dy(2:end-1,:);

                    obj.memory_dvx_dy_gpu(2:end-1,1:end-1) = b_y_slice' .* memory_dvx_slice + ...
                                                        obj.a_y_half_gpu(1:end-1)' .* dvx_dy_slice;

                    % 计算最终值
                    dvy_dx_final = dvy_dx_slice ./ obj.K_x_gpu(2:end-1) + obj.memory_dvy_dx_gpu(2:end-1,1:end-1);
                    dvx_dy_final = dvx_dy_slice ./ obj.K_y_half_gpu(1:end-1)' + obj.memory_dvx_dy_gpu(2:end-1,1:end-1);

                    % 更新剪应力
                    obj.sigmaxy_gpu(2:end-1,1:end-1) = obj.sigmaxy_gpu(2:end-1,1:end-1) + ...
                        obj.c44_gpu(2:end-1,1:end-1) .* (dvy_dx_final + dvx_dy_final) * dt;
                    
                    fprintf('剪应力计算完成，用时: %.3f秒\n', toc(t2));
                    
                    % 计算速度场
                    t3 = tic;
                    fprintf('计算速度场...\n');
                    
                    % x方向速度场
                    dsigmaxx_dx = diff(obj.sigmaxx_gpu, 1, 1) / dx;  % [NX-1, NY]
                    dsigmaxy_dy = diff(obj.sigmaxy_gpu, 1, 2) / dy;  % [NX, NY-1]
                    
                    % 打印原始维度
                    fprintf('原始数组维度:\n');
                    fprintf('dsigmaxx_dx: [%d, %d]\n', size(dsigmaxx_dx,1), size(dsigmaxx_dx,2));
                    fprintf('dsigmaxy_dy: [%d, %d]\n', size(dsigmaxy_dy,1), size(dsigmaxy_dy,2));
                    
                    % 更新记忆变量，使用一致的切片范围
                    obj.memory_dsigmaxx_dx_gpu(2:end-1,:) = obj.b_x_gpu(2:end-1) .* obj.memory_dsigmaxx_dx_gpu(2:end-1,:) + ...
                                                        obj.a_x_gpu(2:end-1) .* dsigmaxx_dx(1:end-1,:);
                    obj.memory_dsigmaxy_dy_gpu(:,2:end-1) = obj.b_y_gpu(2:end-1)' .* obj.memory_dsigmaxy_dy_gpu(:,2:end-1) + ...
                                                        obj.a_y_gpu(2:end-1)' .* dsigmaxy_dy(:,1:end-1);
                    
                    % 计算最终值，确保维度匹配
                    dsigmaxx_dx_final = dsigmaxx_dx(1:end-1,:) ./ obj.K_x_gpu(2:end-1) + ...
                                        obj.memory_dsigmaxx_dx_gpu(2:end-1,:);  % [NX-2, NY]
                    dsigmaxy_dy_final = dsigmaxy_dy(2:end-1,1:end-1) ./ obj.K_y_gpu(2:end-1)' + ...
                                        obj.memory_dsigmaxy_dy_gpu(2:end-1,2:end-1);  % [NX-2, NY-2]
                    
                    % 打印最终维度
                    fprintf('最终数组维度:\n');
                    fprintf('dsigmaxx_dx_final: [%d, %d]\n', size(dsigmaxx_dx_final,1), size(dsigmaxx_dx_final,2));
                    fprintf('dsigmaxy_dy_final: [%d, %d]\n', size(dsigmaxy_dy_final,1), size(dsigmaxy_dy_final,2));
                    
                    % 确保两个数组维度完全匹配后再相加
                    dsigmaxx_dx_final = dsigmaxx_dx_final(:,2:end-1);  % 调整为 [NX-2, NY-2]
                    
                    % 更新速度场
                    obj.vx_gpu(2:end-1,2:end-1) = obj.vx_gpu(2:end-1,2:end-1) + ...
                        (dsigmaxx_dx_final + dsigmaxy_dy_final) .* dt ./ obj.rho_gpu(2:end-1,2:end-1);
                    
                    % y方向速度场
                    dsigmaxy_dx = diff(obj.sigmaxy_gpu, 1, 1) / dx;
                    dsigmayy_dy = diff(obj.sigmayy_gpu, 1, 2) / dy;
                    
                    obj.memory_dsigmaxy_dx_gpu(1:end-1,:) = obj.b_x_half_gpu(1:end-1) .* obj.memory_dsigmaxy_dx_gpu(1:end-1,:) + ...
                                                obj.a_x_half_gpu(1:end-1) .* dsigmaxy_dx;
                    obj.memory_dsigmayy_dy_gpu(:,1:end-1) = obj.b_y_half_gpu(1:end-1)' .* obj.memory_dsigmayy_dy_gpu(:,1:end-1) + ...
                                                obj.a_y_half_gpu(1:end-1)' .* dsigmayy_dy;
                    
                    dsigmaxy_dx_final = dsigmaxy_dx ./ obj.K_x_half_gpu(1:end-1) + obj.memory_dsigmaxy_dx_gpu(1:end-1,:);
                    dsigmayy_dy_final = dsigmayy_dy ./ obj.K_y_half_gpu(1:end-1)' + obj.memory_dsigmayy_dy_gpu(:,1:end-1);
                    
                    obj.vy_gpu(1:end-1,1:end-1) = obj.vy_gpu(1:end-1,1:end-1) + ...
                        (dsigmaxy_dx_final + dsigmayy_dy_final) .* dt ./ obj.rho_gpu(1:end-1,1:end-1);
                    
                    fprintf('速度场计算完成，用时: %.3f秒\n', toc(t3));
                    
                    % GPU内存使用情况检查
                    gpu_info = gpuDevice();
                    fprintf('\nGPU内存使用情况:\n');
                    fprintf('总内存: %.2f GB\n', gpu_info.TotalMemory/1e9);
                    fprintf('可用内存: %.2f GB\n', gpu_info.AvailableMemory/1e9);
                    fprintf('已用内存: %.2f GB\n', (gpu_info.TotalMemory - gpu_info.AvailableMemory)/1e9);
                    
                    compute_time = toc(t_start);
                    fprintf('\n总计算用时: %.3f秒\n', compute_time);
                    
                    % 检查数值稳定性
                    if any(isnan(gather(obj.vx_gpu(:)))) || any(isinf(gather(obj.vx_gpu(:))))
                        warning('x方向速度场中出现NaN或Inf值！');
                    end
                    if any(isnan(gather(obj.vy_gpu(:)))) || any(isinf(gather(obj.vy_gpu(:))))
                        warning('y方向速度场中出现NaN或Inf值！');
                    end
                end 
        %}


        function apply_boundary_conditions(obj)
            % 应用Dirichlet边界条件（刚性边界）
            % 设置x方向速度边界条件
            obj.vx(1,:) = obj.ZERO;    % 左边界
            obj.vx(end,:) = obj.ZERO;  % 右边界
            obj.vx(:,1) = obj.ZERO;    % 下边界
            obj.vx(:,end) = obj.ZERO;  % 上边界
            
            % 设置y方向速度边界条件
            obj.vy(1,:) = obj.ZERO;    % 左边界
            obj.vy(end,:) = obj.ZERO;  % 右边界
            obj.vy(:,1) = obj.ZERO;    % 下边界
            obj.vy(:,end) = obj.ZERO;  % 上边界
        end
        
        function record_seismograms(obj, it)
            % 在检波器位置记录地震图数据
            vx_data = obj.vx;  % x方向速度场
            vy_data = obj.vy;  % y方向速度场
            
            % 在每个检波器位置记录速度值
            for i = 1:obj.NREC
                obj.seismogram_vx(it, i) = vx_data(obj.rec_x(i), obj.rec_y(i));
                obj.seismogram_vy(it, i) = vy_data(obj.rec_x(i), obj.rec_y(i));
            end
        end
        
        
        %{
        function record_seismograms_gpu(obj, it)
                    % GPU版本的地震图记录函数
                    % 直接使用GPU上的数据，不需要传回CPU
                    for i = 1:obj.NREC
                        obj.seismogram_vx(it, i) = obj.vx_gpu(obj.rec_x(i), obj.rec_y(i));
                        obj.seismogram_vy(it, i) = obj.vy_gpu(obj.rec_y(i), obj.rec_y(i));
                    end
                end 
        %}

        
        function output_info(obj, it)
            % 输出模拟状态信息和保存波场数据
            vx_data = obj.vx;
            vy_data = obj.vy;
            
            % 计算速度场的最大幅值
            velocnorm = max(max(sqrt(vx_data.^2 + vy_data.^2)));
            
            % 输出当前时间步和炮号信息（统一格式，不区分单炮多炮）
            fprintf('炮号: %d/%d, 时间步: %d/%d\n', ...
                obj.current_shot_number, obj.NSHOT, it, obj.NSTEP);
            fprintf('模拟时间: %.6f 秒\n', (it-1)*obj.DELTAT);
            fprintf('速度矢量最大范数 (m/s) = %f\n\n', velocnorm);
            
            % 检查数值稳定性
            if velocnorm > obj.STABILITY_THRESHOLD
                error('模拟变得不稳定并发散');
            end
            
            % 根据参数控制是否保存波场快照
            if obj.save_snapshots
                % 创建以炮号命名的子目录
                shot_dir = fullfile(obj.output_dir, sprintf('shot_%03d', obj.current_shot_number));
                if ~exist(shot_dir, 'dir')
                    mkdir(shot_dir);
                end
                
                % 保存波场数据
                save_path = fullfile(shot_dir, sprintf('wavefield_%06d.mat', it));
                save(save_path, 'vx_data', 'vy_data', '-v7.3');
            end
        end
        
        
        %{
        function output_info_gpu(obj, it)
                    % GPU版本的输出信息函数
                    % 计算速度场的最大幅值（在GPU上进行）
                    velocnorm = max(max(sqrt(obj.vx_gpu.^2 + obj.vy_gpu.^2)));
                    velocnorm = gather(velocnorm);  % 只传输一个标量值
                    
                    % 输出信息
                    fprintf('炮号: %d/%d, 时间步: %d/%d\n', ...
                        obj.current_shot_number, obj.NSHOT, it, obj.NSTEP);
                    fprintf('模拟时间: %.6f 秒\n', (it-1)*obj.DELTAT);
                    fprintf('速度矢量最大范数 (m/s) = %f\n\n', velocnorm);
                    
                    % 检查数值稳定性
                    if velocnorm > obj.STABILITY_THRESHOLD
                        error('模拟变得不稳定并发散');
                    end
                    
                    % 保存波场快照（如果需要）
                    if obj.save_snapshots
                        % 创建目录
                        shot_dir = fullfile(obj.output_dir, sprintf('shot_%03d', obj.current_shot_number));
                        if ~exist(shot_dir, 'dir')
                            mkdir(shot_dir);
                        end
                        
                        % 保存GPU数据
                        vx_data = obj.vx_gpu;  % 保持在GPU上
                        vy_data = obj.vy_gpu;
                        save_path = fullfile(shot_dir, sprintf('wavefield_%06d.mat', it));
                        save(save_path, 'vx_data', 'vy_data', '-v7.3');
                    end
                end 
        %}

        
        function reset_fields(obj)
            % 重置波场和记忆变量
            obj.vx = zeros(obj.NX, obj.NY);
            obj.vy = zeros(obj.NX, obj.NY);
            obj.sigmaxx = zeros(obj.NX, obj.NY);
            obj.sigmayy = zeros(obj.NX, obj.NY);
            obj.sigmaxy = zeros(obj.NX, obj.NY);
            
            % 重置PML记忆变量
            obj.memory_dvx_dx = zeros(obj.NX, obj.NY);
            obj.memory_dvx_dy = zeros(obj.NX, obj.NY);
            obj.memory_dvy_dx = zeros(obj.NX, obj.NY);
            obj.memory_dvy_dy = zeros(obj.NX, obj.NY);
            obj.memory_dsigmaxx_dx = zeros(obj.NX, obj.NY);
            obj.memory_dsigmayy_dy = zeros(obj.NX, obj.NY);
            obj.memory_dsigmaxy_dx = zeros(obj.NX, obj.NY);
            obj.memory_dsigmaxy_dy = zeros(obj.NX, obj.NY);
        end
    end
end 