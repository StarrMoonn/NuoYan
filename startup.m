% 添加所有必要的路径
project_root = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(project_root, 'core')));
addpath(genpath(fullfile(project_root, 'mex')));
addpath(genpath(fullfile(project_root, 'unit_test'))); 