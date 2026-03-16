function truth = DGP_wave_equation(cfg, m, c_wave, delta_t_wave, sigma2_v)
%%% DGP_wave_equation %%%
%
% Jilles van Hulst
%
% 10-11-2023  (refactored 2026)
%
%
% Description:
%   Generate ground-truth data from D'Alembert's solution of the 1-D wave
%   equation with initial condition m(x) and zero initial velocity:
%       f(x,t) = [m(x - c*t) + m(x + c*t)] / 2
%
% Inputs:
%   cfg          struct with fields .p, .N, .N_test, .x_min, .x_max, .x_test
%   m            initial condition function handle  @(X) -> (N x 1)
%   c_wave       wave speed  (scalar)
%   delta_t_wave sample time  (scalar)
%   sigma2_v     measurement noise variance  (scalar)
%
% Outputs:
%   truth     struct with fields:
%               .x       (p x N)       measurement locations
%               .y       (p x N)       noisy measurements
%               .f_true  (N_test x N)  true function at test points


%% Unpack configuration

p      = cfg.p;
N      = cfg.N;
N_test = cfg.N_test;
x_min  = cfg.x_min;
x_max  = cfg.x_max;
x_test = cfg.x_test;

Sigma2_v = sigma2_v * eye(p);


%% Initialize

x      = NaN(p, N);
y      = NaN(p, N);
f_true = NaN(N_test, N);


%% Simulate

for t = 1:N

    % random measurement locations:
    x(:,t) = unifrnd(x_min, x_max, p, 1);

    % evaluate D'Alembert solution (t-1 so first snapshot is the i.c.):
    time = delta_t_wave * (t - 1);
    f_true(:,t)  = wave_eq(x_test, m, c_wave, time);    % test grid
    f_samp       = wave_eq(x(:,t), m, c_wave, time);    % sample points

    % noisy measurement:
    v      = mvnrnd(zeros(p,1), Sigma2_v).';
    y(:,t) = f_samp + v;

end


%% Pack outputs

truth.x      = x;
truth.y      = y;
truth.f_true = f_true;

end


%% ==================  Local Functions  ==================================

function f = wave_eq(x, m, c, time)
    % D'Alembert formula (zero initial velocity)
    f = (m(x - c*time) + m(x + c*time)) / 2;
end
