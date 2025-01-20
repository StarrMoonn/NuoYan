function [vx, vy] = compute_wave_propagation_cpu1(obj)
    % 从对象中获取计算参数
    dx = obj.DELTAX;
    dy = obj.DELTAY;
    dt = obj.DELTAT;
    
    % 获取当前状态变量的引用（为了代码简洁性）
    vx = obj.vx;
    vy = obj.vy;
    sigmaxx = obj.sigmaxx;
    sigmayy = obj.sigmayy;
    sigmaxy = obj.sigmaxy;
    
    % 计算应力场 sigmaxx 和 sigmayy
    for j = 2:obj.NY
        for i = 1:obj.NX-1
            % 计算速度梯度
            value_dvx_dx = (vx(i+1,j) - vx(i,j)) / dx;
            value_dvy_dy = (vy(i,j) - vy(i,j-1)) / dy;
            
            % 更新PML记忆变量
            obj.memory_dvx_dx(i,j) = obj.b_x_half(i) * obj.memory_dvx_dx(i,j) + ...
                                    obj.a_x_half(i) * value_dvx_dx;
            obj.memory_dvy_dy(i,j) = obj.b_y(j) * obj.memory_dvy_dy(i,j) + ...
                                    obj.a_y(j) * value_dvy_dy;
            
            % 计算最终值
            value_dvx_dx = value_dvx_dx / obj.K_x_half(i) + obj.memory_dvx_dx(i,j);
            value_dvy_dy = value_dvy_dy / obj.K_y(j) + obj.memory_dvy_dy(i,j);
            
            % 更新应力
            sigmaxx(i,j) = sigmaxx(i,j) + dt * (...
                obj.c11(i,j) * value_dvx_dx + ...
                obj.c13(i,j) * value_dvy_dy);
            
            sigmayy(i,j) = sigmayy(i,j) + dt * (...
                obj.c13(i,j) * value_dvx_dx + ...
                obj.c33(i,j) * value_dvy_dy);
        end
    end
    
    % 计算剪应力 sigmaxy
    for j = 1:obj.NY-1
        for i = 2:obj.NX
            value_dvy_dx = (vy(i,j) - vy(i-1,j)) / dx;
            value_dvx_dy = (vx(i,j+1) - vx(i,j)) / dy;
            
            obj.memory_dvy_dx(i,j) = obj.b_x(i) * obj.memory_dvy_dx(i,j) + ...
                                    obj.a_x(i) * value_dvy_dx;
            obj.memory_dvx_dy(i,j) = obj.b_y_half(j) * obj.memory_dvx_dy(i,j) + ...
                                    obj.a_y_half(j) * value_dvx_dy;
            
            value_dvy_dx = value_dvy_dx / obj.K_x(i) + obj.memory_dvy_dx(i,j);
            value_dvx_dy = value_dvx_dy / obj.K_y_half(j) + obj.memory_dvx_dy(i,j);
            
            sigmaxy(i,j) = sigmaxy(i,j) + ...
                obj.c44(i,j) * (value_dvy_dx + value_dvx_dy) * dt;
        end
    end
    
    % 计算x方向速度场
    for j = 2:obj.NY
        for i = 2:obj.NX
            value_dsigmaxx_dx = (sigmaxx(i,j) - sigmaxx(i-1,j)) / dx;
            value_dsigmaxy_dy = (sigmaxy(i,j) - sigmaxy(i,j-1)) / dy;
            
            obj.memory_dsigmaxx_dx(i,j) = obj.b_x(i) * obj.memory_dsigmaxx_dx(i,j) + ...
                                         obj.a_x(i) * value_dsigmaxx_dx;
            obj.memory_dsigmaxy_dy(i,j) = obj.b_y(j) * obj.memory_dsigmaxy_dy(i,j) + ...
                                         obj.a_y(j) * value_dsigmaxy_dy;
            
            value_dsigmaxx_dx = value_dsigmaxx_dx / obj.K_x(i) + obj.memory_dsigmaxx_dx(i,j);
            value_dsigmaxy_dy = value_dsigmaxy_dy / obj.K_y(j) + obj.memory_dsigmaxy_dy(i,j);
            
            vx(i,j) = vx(i,j) + ...
                (value_dsigmaxx_dx + value_dsigmaxy_dy) * dt / obj.rho(i,j);
        end
    end
    
    % 计算y方向速度场
    for j = 1:obj.NY-1
        for i = 1:obj.NX-1
            value_dsigmaxy_dx = (sigmaxy(i+1,j) - sigmaxy(i,j)) / dx;
            value_dsigmayy_dy = (sigmayy(i,j+1) - sigmayy(i,j)) / dy;
            
            obj.memory_dsigmaxy_dx(i,j) = obj.b_x_half(i) * obj.memory_dsigmaxy_dx(i,j) + ...
                                         obj.a_x_half(i) * value_dsigmaxy_dx;
            obj.memory_dsigmayy_dy(i,j) = obj.b_y_half(j) * obj.memory_dsigmayy_dy(i,j) + ...
                                         obj.a_y_half(j) * value_dsigmayy_dy;
            
            value_dsigmaxy_dx = value_dsigmaxy_dx / obj.K_x_half(i) + obj.memory_dsigmaxy_dx(i,j);
            value_dsigmayy_dy = value_dsigmayy_dy / obj.K_y_half(j) + obj.memory_dsigmayy_dy(i,j);
            
            vy(i,j) = vy(i,j) + ...
                (value_dsigmaxy_dx + value_dsigmayy_dy) * dt / obj.rho(i,j);
        end
    end
end 