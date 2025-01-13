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
        compute_kernel    % 'cpu' 或 'cpu_mex' 或 'cuda_mex'
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
                    obj.compute_wave_propagation_cpu1();  % 遍历 CPU实现
                case 'cpu_mex'
                    obj.compute_wave_propagation_cpu2();  % C++ CPU实现
                case 'cuda_mex'
                    obj.compute_wave_propagation_gpu();   % CUDA GPU实现
                otherwise
                    error('Unknown compute kernel type: %s', obj.compute_kernel);
            end
        end

        function compute_wave_propagation_cpu1(obj)
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

        function compute_wave_propagation_cpu2(obj)
            % 使用类成员变量
            dx = obj.DELTAX;
            dy = obj.DELTAY;
            dt = obj.DELTAT;
            
            % 将所需的数据准备好，并传递给 MEX 函数
            vx = obj.vx;           % x方向速度场
            vy = obj.vy;           % y方向速度场
            sigmaxx = obj.sigmaxx; % x方向应力
            sigmayy = obj.sigmayy; % y方向应力
            sigmaxy = obj.sigmaxy; % 剪切应力

            % 内存变量
            memory_dvx_dx = obj.memory_dvx_dx; 
            memory_dvy_dy = obj.memory_dvy_dy;
            memory_dvy_dx = obj.memory_dvy_dx;
            memory_dvx_dy = obj.memory_dvx_dy;
            memory_dsigmaxx_dx = obj.memory_dsigmaxx_dx;
            memory_dsigmaxy_dy = obj.memory_dsigmaxy_dy;
            memory_dsigmaxy_dx = obj.memory_dsigmaxy_dx;
            memory_dsigmayy_dy = obj.memory_dsigmayy_dy;

            % 材料参数
            c11 = obj.c11;
            c13 = obj.c13;
            c33 = obj.c33;
            c44 = obj.c44;
            rho = obj.rho;

            % PML参数
            b_x = obj.b_x;
            b_y = obj.b_y;
            b_x_half = obj.b_x_half;
            b_y_half = obj.b_y_half;

            a_x = obj.a_x;
            a_y = obj.a_y;
            a_x_half = obj.a_x_half;
            a_y_half = obj.a_y_half;

            K_x = obj.K_x;
            K_y = obj.K_y;
            K_x_half = obj.K_x_half;
            K_y_half = obj.K_y_half;

            % 计算参数
            DELTAX = obj.DELTAX;
            DELTAY = obj.DELTAY;
            DELTAT = obj.DELTAT;
            NX = obj.NX;
            NY = obj.NY;

            % 调用编译好的 MEX 文件
            [vx, vy] = compute_wave_propagation(vx, vy, sigmaxx, sigmayy, sigmaxy, ...
                memory_dvx_dx, memory_dvy_dy, memory_dvy_dx, memory_dvx_dy, ...
                memory_dsigmaxx_dx, memory_dsigmaxy_dy, memory_dsigmaxy_dx, memory_dsigmayy_dy, ...
                c11, c13, c33, c44, rho, ...
                b_x, b_y, b_x_half, b_y_half, ...
                a_x, a_y, a_x_half, a_y_half, ...
                K_x, K_y, K_x_half, K_y_half, ...
                DELTAX, DELTAY, DELTAT, NX, NY);

            % 将 MEX 输出结果赋值给类成员变量
            obj.vx = vx;
            obj.vy = vy;
        end
        
        function compute_wave_propagation_gpu(obj)
            % Similar to compute_wave_propagation_cpu2 but using GPU version
            dx = obj.DELTAX;
            dy = obj.DELTAY;
            dt = obj.DELTAT;
            
            % Prepare data for MEX function
            vx = obj.vx;           
            vy = obj.vy;           
            sigmaxx = obj.sigmaxx; 
            sigmayy = obj.sigmayy; 
            sigmaxy = obj.sigmaxy; 

            % Memory variables
            memory_dvx_dx = obj.memory_dvx_dx; 
            memory_dvy_dy = obj.memory_dvy_dy;
            memory_dvy_dx = obj.memory_dvy_dx;
            memory_dvx_dy = obj.memory_dvx_dy;
            memory_dsigmaxx_dx = obj.memory_dsigmaxx_dx;
            memory_dsigmaxy_dy = obj.memory_dsigmaxy_dy;
            memory_dsigmaxy_dx = obj.memory_dsigmaxy_dx;
            memory_dsigmayy_dy = obj.memory_dsigmayy_dy;

            % Material parameters
            c11 = obj.c11;
            c13 = obj.c13;
            c33 = obj.c33;
            c44 = obj.c44;
            rho = obj.rho;

            % PML parameters
            b_x = obj.b_x;
            b_y = obj.b_y;
            b_x_half = obj.b_x_half;
            b_y_half = obj.b_y_half;
            a_x = obj.a_x;
            a_y = obj.a_y;
            a_x_half = obj.a_x_half;
            a_y_half = obj.a_y_half;
            K_x = obj.K_x;
            K_y = obj.K_y;
            K_x_half = obj.K_x_half;
            K_y_half = obj.K_y_half;

            % Computation parameters
            DELTAX = obj.DELTAX;
            DELTAY = obj.DELTAY;
            DELTAT = obj.DELTAT;
            NX = obj.NX;
            NY = obj.NY;

            % Call the GPU MEX file
            %[vx, vy] =  VTI_WaveFieldSolver_SIMD
            %[vx, vy] = compute_wave_propagation_omp
            [vx, vy] =  compute_wave_propagation_omp(vx, vy, sigmaxx, sigmayy, sigmaxy, ...
                memory_dvx_dx, memory_dvy_dy, memory_dvy_dx, memory_dvx_dy, ...
                memory_dsigmaxx_dx, memory_dsigmaxy_dy, memory_dsigmaxy_dx, memory_dsigmayy_dy, ...
                c11, c13, c33, c44, rho, ...
                b_x, b_y, b_x_half, b_y_half, ...
                a_x, a_y, a_x_half, a_y_half, ...
                K_x, K_y, K_x_half, K_y_half, ...
                DELTAX, DELTAY, DELTAT, NX, NY);

            % Update only velocity fields
            obj.vx = vx;
            obj.vy = vy;
        end

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