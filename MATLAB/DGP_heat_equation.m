function truth = DGP_heat_equation(cfg, alpha, delta_t, sigma2_v)
%%% DGP_heat_equation %%%
%
% Jilles van Hulst
%
% 10-11-2023  (refactored 2026)
%
%
% Description:
%   Generate ground-truth data from the analytical solution of the 1-D heat
%   equation with a Dirac-delta initial condition on the real line.
%   The solution is the Gaussian Green's function:
%       f(x,t) = 1/sqrt(4*pi*alpha*t) * exp(-x^2 / (4*alpha*t))
%
% Inputs:
%   cfg       struct with fields .p, .N, .N_test, .x_min, .x_max, .x_test
%   alpha     thermal diffusivity  (scalar)
%   delta_t   sample time          (scalar)
%   sigma2_v  measurement noise variance  (scalar)
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

    % evaluate analytical solution:
    f_true(:,t)  = heat_eq(x_test, alpha, delta_t * t);   % test grid
    f_samp       = heat_eq(x(:,t), alpha, delta_t * t);   % sample points

    % noisy measurement:
    v     = mvnrnd(zeros(p,1), Sigma2_v).';
    y(:,t) = f_samp + v;

end


%% Pack outputs

truth.x      = x;
truth.y      = y;
truth.f_true = f_true;

end


%% ==================  Local Functions  ==================================

function f = heat_eq(x, alpha, time)
    % Gaussian Green's function (Dirac i.c.)
    f = 1/sqrt(4*pi*alpha*time) * exp(-(x.^2) / (4*alpha*time));
end
