function [truth, cfg] = DGP_data(cfg)
%%% DGP_data %%%
%
% Jilles van Hulst
%
% 10-11-2023  (refactored 2026)
%
%
% Description:
%   Load ground-truth data from a .mat file.  The loaded data may
%   overwrite cfg fields (N, p, x_min, x_max, N_test, x_test) to match
%   the dataset.
%
% Expected .mat contents:
%   x  (p x N)       measurement locations
%   y  (p x N)       noisy measurements
%
%   and optionally one of:
%     f       function handle   f(x_test) -> (N_test x N)
%     f_true  (N_test x N)     precomputed true function
%     x_test  (N_test x 1)     if f_true's grid differs from cfg.x_test
%
% Inputs:
%   cfg   struct (see Dynamic_Gaussian_Process_Main for fields)
%
% Outputs:
%   truth struct with fields .x, .y, .f_true
%   cfg   struct — potentially updated fields (N, p, x_min, x_max, ...)


%% Load data

load('heat_equation_simulation.mat');


%% Overwrite config from data

cfg.N = size(y, 2);
cfg.p = size(y, 1);

if exist('f', 'var')
    f_true = f(cfg.x_test);
elseif exist('f_true', 'var') && isequal(size(cfg.x_test,1), size(f_true,1))
    cfg.x_min  = min(x_test);
    cfg.x_max  = max(x_test);
    cfg.N_test = size(x_test, 1);
    cfg.x_test = x_test;
    cfg.dx     = (cfg.x_max - cfg.x_min) / (cfg.N_test - 1);
elseif ~exist('f_true', 'var')
    f_true = NaN(cfg.N_test, cfg.N);
else
    error('Dataset does not have the correct format')
end


%% Pack outputs

truth.x      = x;
truth.y      = y;
truth.f_true = f_true;

end
