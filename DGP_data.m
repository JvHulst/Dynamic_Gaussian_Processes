%%% DGP_data %%%
% 
% 
% Jilles van Hulst
% 
% 10-11-2023
% 
% 
% Description:
% Generation of the 'true' function data for DGP estimator based on a
% dataset of measurement locations and corresponding noisy function
% observations.



%% Load data


% loaded data must have format/names
% x: [p by N]                           measurement locations for every timestep
% y: [p by N]                           measurements for every timestep
%
% with optionally: 
% f: \mathcal{X}^N \to \mathbb{R}^N     true function which takes vector inputs
% OR
% x_test:  [N_test by 1]                  evaluation test points
% f_true: [N_test by N]                  true function evaluated at x_test for very timestep

load('heat_equation_simulation_no_noise.mat');


%% Overwrite variables based on loaded dataset


N = size(y,2);
p = size(y,1);

if exist('f','var')
    f_true = f(x_test);
elseif exist('f_true','var') && exist('x_test','var')
    x_min = min(x_test);
    x_max = max(x_test);
    N_test = size(x_test,1);
elseif ~exist('f_true','var') && ~exist('x_test','var')
    f_true = NaN(N_test,N);
else
    error('Dataset does not have the correct format')
end


