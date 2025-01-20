function [vx, vy] = compute_wave_propagation_cpu4(obj)
    % 使用 SIMD (AVX2) 和 OpenMP 优化的波场传播计算
    % 从对象中获取所需参数
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

    % 调用编译好的 SIMD+OpenMP MEX 文件
    [vx, vy] = VTI_WaveFieldSolver_SIMD(vx, vy, sigmaxx, sigmayy, sigmaxy, ...
        memory_dvx_dx, memory_dvy_dy, memory_dvy_dx, memory_dvx_dy, ...
        memory_dsigmaxx_dx, memory_dsigmaxy_dy, memory_dsigmaxy_dx, memory_dsigmayy_dy, ...
        c11, c13, c33, c44, rho, ...
        b_x, b_y, b_x_half, b_y_half, ...
        a_x, a_y, a_x_half, a_y_half, ...
        K_x, K_y, K_x_half, K_y_half, ...
        DELTAX, DELTAY, DELTAT, NX, NY);
end 