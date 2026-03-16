function truth = DGP_simulation(cfg, kf, m, Q_f, Q_w, Q_v)
%%% DGP_simulation %%%
%
% Jilles van Hulst
%
% 10-11-2023  (refactored 2026)
%
%
% Description:
%   Generate ground-truth data by simulating the DGP system equations using
%   a fine discrete-bin approximation.  The 'true' system is represented by
%   N_test discrete basis functions.
%
%   NOTE: this truth generator only supports single-state systems.  For
%   multi-state systems (e.g. wave equation), use an analytical truth
%   generator instead.
%
% Inputs:
%   cfg   struct with fields .p, .N, .N_test, .x_min, .x_max, .x_test
%   kf    evolution kernel    — function handle @(X,V)  (NOT cell array)
%   m     initial mean        — function handle @(X)
%   Q_f   initial covariance  — function handle @(X,V)
%   Q_w   disturbance cov.    — function handle @(X,V)
%   Q_v   measurement noise   — function handle @(X,V)
%
% Outputs:
%   truth     struct with fields:
%               .x       (p x N)       measurement locations
%               .y       (p x N)       noisy measurements
%               .f_true  (N_test x N)  true function at test points


%% Guard: multi-state not supported

if iscell(kf)
    error(['DGP_simulation does not support multi-state systems ', ...
           '(cell-array kf).  Use an analytical truth generator instead.'])
end


%% Unpack configuration

p      = cfg.p;
N      = cfg.N;
N_test = cfg.N_test;
x_min  = cfg.x_min;
x_max  = cfg.x_max;
x_test = cfg.x_test;


%% Approximate the system functions using M_true discrete bins

M_true     = N_test;
dx_fit     = (x_max - x_min) / (M_true - 1);
x_fit      = (x_min : dx_fit : x_max).';

bounds     = x_min : (x_max*(1+1e-12) - x_min) / M_true : x_max*(1+1e-12);

u_true     = cell(M_true, 1);
U_fit      = NaN(M_true);
U_test     = NaN(M_true, N_test);
U_samp     = NaN(M_true, p);

for i = 1:M_true
    u_true{i}    = @(X) double(bounds(i) <= X & X < bounds(i+1)) / dx_fit;
    U_fit(i,:)   = u_true{i}(x_fit);
    U_test(i,:)  = u_true{i}(x_test);
end

% fitting (simple since orthonormal):
Pinv_U = (U_fit.' * U_fit) \ U_fit.';

Lambda_true = Pinv_U * kf(x_fit, x_fit);
z_bar_true  = Pinv_U * m(x_fit);
Lambda_f    = Pinv_U * Q_f(x_fit, x_fit);
Lambda_w    = Pinv_U * Q_w(x_fit, x_fit);

Lambda_U    = U_fit.' * U_fit * dx_fit^2;

% symmetrise covariance matrices:
Lambda_f = (Lambda_f.' + Lambda_f) / 2;
Lambda_w = (Lambda_w.' + Lambda_w) / 2;

% stability check:
eig_max = max(abs(eig(Lambda_true * Lambda_U)));
if eig_max < 1
    fprintf('''True'' system is stable (max |eig| = %.4f)\n', eig_max)
elseif abs(eig_max - 1) < 1e-6
    fprintf('''True'' system is marginally stable (max |eig| = %.6f)\n', eig_max)
else
    fprintf('''True'' system is NOT stable (max |eig| = %.4f)\n', eig_max)
end


%% Initialize

x      = NaN(p, N);
y      = NaN(p, N);
z      = NaN(M_true, N+1);
f_true = NaN(N_test, N);

% draw initial state from prior:
z(:,1) = mvnrnd(z_bar_true, Lambda_f).';


%% Simulate

for t = 1:N

    % random measurement locations:
    x(:,t) = unifrnd(x_min, x_max, p, 1);

    % evaluate basis functions at sample points:
    for i = 1:M_true
        U_samp(i,:) = u_true{i}(x(:,t));
    end

    % true function:
    f_true(:,t) = U_test.' * z(:,t);
    f_samp      = U_samp.' * z(:,t);

    % noisy measurement:
    v      = mvnrnd(zeros(p,1), Q_v(x(:,t), x(:,t))).';
    y(:,t) = f_samp + v;

    % dynamics:
    xi       = mvnrnd(zeros(M_true,1), Lambda_w).';
    z(:,t+1) = Lambda_true * Lambda_U * z(:,t) + xi;

end


%% Pack outputs

truth.x      = x;
truth.y      = y;
truth.f_true = f_true;

end
