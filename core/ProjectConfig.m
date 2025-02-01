classdef ProjectConfig
    properties (Constant)
        % 获取项目根目录
        PROJECT_ROOT = fileparts(fileparts(mfilename('fullpath')));
        
        % 定义所有输出目录
        OUTPUT_DIR = fullfile(ProjectConfig.PROJECT_ROOT, 'data', 'output');
        WAVEFIELD_DIR = fullfile(ProjectConfig.OUTPUT_DIR, 'wavefield');
        FWI_DIR = fullfile(ProjectConfig.OUTPUT_DIR, 'fwi');
        
        % FWI相关目录
        FWI_MISFIT_DIR = fullfile(ProjectConfig.FWI_DIR, 'misfit');
        FWI_GRADIENT_DIR = fullfile(ProjectConfig.FWI_DIR, 'gradient');
        
        % 波场相关目录
        ADJOINT_WAVEFIELD_DIR = fullfile(ProjectConfig.WAVEFIELD_DIR, 'adjoint');
    end
    
    methods (Static)
        function create_directories()
            % 创建所有必要的目录
            dirs = {
                ProjectConfig.OUTPUT_DIR, 
                ProjectConfig.WAVEFIELD_DIR,
                ProjectConfig.FWI_DIR,
                ProjectConfig.FWI_MISFIT_DIR,
                ProjectConfig.FWI_GRADIENT_DIR,
                ProjectConfig.ADJOINT_WAVEFIELD_DIR
            };
            
            for i = 1:length(dirs)
                if ~exist(dirs{i}, 'dir')
                    mkdir(dirs{i});
                end
            end
        end
    end
end 