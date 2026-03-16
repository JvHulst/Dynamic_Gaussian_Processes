%%% Dynamic_Gaussian_Processes %%%
% 
% 
% Jilles van Hulst
% 
% 17-02-2023  (refactored 2026)
% 
% 
% Description:
% Matlab implementation accompanying the paper
%
% "Estimation of Dynamic Gaussian Processes"
%
% Authors:
% Jilles van Hulst
% Roy van Zuijlen
% Duarte Antunes
% W.P.M.H. (Maurice) Heemels
%
% The methodology used in this script can be simultaneously considered
% 1) An extension of Gaussian processes for evolving functions.
% 2) An extension of Kalman filter for infinite-dimensional states (functions).
% This script considers an approximate solution using basis functions which
% allows for recursive and fast computation. The script automatically
% approximates user-defined system functions.
% This script considers the case when \mathcal{X} is a closed interval in
% \mathbb{R}.
%
% There are multiple options for the generation of the 'true' function data
% 1) From a discretized approximation of the system functions (DGP_simulation)
% 2) From the solution to the 1D heat equation PDE (DGP_heat_equation)
% 3) From data (DGP_data)
% 4) From the solution to the 1D wave equation PDE (DGP_wave_equation)

close all
clear; clc


set(0,'defaultTextInterpreter','latex');


%% Configuration

cfg.plot_confidence_bounds = true;     % plot covariance estimates
cfg.plot_basis_functions   = false;    % plot basis functions

cfg.N_test = 251;                      % evaluation grid resolution
cfg.x_min  = -10;                      % domain lower bound
cfg.x_max  =  10;                      % domain upper bound
cfg.dx     = (cfg.x_max - cfg.x_min) / (cfg.N_test - 1);
cfg.x_test = (cfg.x_min : cfg.dx : cfg.x_max).';

cfg.p = 5;                             % observations per timestep
cfg.N = 21;                            % number of timesteps

cfg.M = 31;                            % number of basis functions

cfg.basis  = 'Fourier';                % 'RBF' | 'NRBF' | 'Fourier' | 'Discrete'
cfg.system = 'Wave_equation';          % 'Discrete_approximation' | 'Heat_equation' | 'Wave_equation' | 'Data'

cfg.kernel       = 'wave_equation';    % 'heat_equation' | 'wave_equation' | 'smoothing' | 'integrator'
cfg.initial_mean = 'squexp';           % 'zero' | 'squexp' | 'ones' | 'parabola' | 'dirac'


%% System Functions

% helper for zero kernel blocks (used by multi-state systems):
zero_kernel = @(X,V) zeros(size(X(:),1), size(V(:),1));


%%% ── SELECT evolution kernel (kf) ──────────────────────────────────
% Controlled by cfg.kernel.  Each case builds kf as either a single
% function handle (scalar state) or a 2x2 cell array (multi-state).

switch cfg.kernel
    case 'integrator'
        % Kronecker-delta kernel: the function is preserved across time.
        kf = @(X,V) double(abs(X(:)-V(:).') <= cfg.dx/2) / cfg.dx;

    case 'smoothing'
        % Gaussian smoothing kernel: slight diffusion each step.
        sigma2_kf = 0.5;  a_kf = 0.8;
        kf = @(X,V) a_kf * squexp(X,V,sigma2_kf) / sqrt(2*pi) / sigma2_kf;

    case 'heat_equation'
        % Gaussian Green's function of the 1-D heat equation.
        %   alpha   : thermal diffusivity  (m^2/s)
        %   delta_t : sample time between snapshots (s)
        alpha   = 1;
        delta_t = 1;
        kf = @(X,V) 1/sqrt(4*pi*alpha*delta_t) ...
            * exp(-(X(:)-V(:).').^2 / (4*alpha*delta_t));

    case 'wave_equation'
        % D'Alembert-based 2x2 kernel for the 1-D wave equation.
        %   c_wave       : wave speed  (m/s)
        %   delta_t_wave : sample time between snapshots (s)
        %   eps_wave     : Gaussian approximation width (= grid spacing)
        c_wave       = 2;
        delta_t_wave = 0.2;
        shift        = c_wave * delta_t_wave;
        eps_wave     = cfg.dx;
        kf_ff = @(X,V) ( exp(-(X(:)-V(:).'-shift).^2/(2*eps_wave^2)) ...
                       + exp(-(X(:)-V(:).'+shift).^2/(2*eps_wave^2)) ) ...
                       / (2*eps_wave*sqrt(2*pi));
        kf_fg = @(X,V) 1/(4*c_wave) ...
                * ( erf((V(:).'-X(:)+shift)/(eps_wave*sqrt(2))) ...
                  - erf((V(:).'-X(:)-shift)/(eps_wave*sqrt(2))) );
        kf_gf = @(X,V) c_wave / (2*eps_wave^3*sqrt(2*pi)) * ( ...
            (X(:)-V(:).'-shift) .* exp(-(X(:)-V(:).'-shift).^2/(2*eps_wave^2)) ...
          - (X(:)-V(:).'+shift) .* exp(-(X(:)-V(:).'+shift).^2/(2*eps_wave^2)) );
        kf_gg = @(X,V) kf_ff(X,V);
        kf = {kf_ff, kf_fg; kf_gf, kf_gg};

    otherwise
        error('Unknown cfg.kernel: "%s"', cfg.kernel)
end


%%% ── SELECT initial function mean (m) ──────────────────────────────
% Controlled by cfg.initial_mean.

switch cfg.initial_mean
    case 'zero'
        m = @(X) zeros(size(X,1),1);
    case 'squexp'
        m = @(X) 10*squexp(X,0,1);              % Gaussian bump at origin
    case 'ones'
        m = @(X) ones(size(X,1),1);
    case 'parabola'
        m = @(X) -2*X.^2 + 8;
    case 'dirac'
        m = @(X) double(abs(X(:))<=0.05)/0.1;   % narrow pulse
    otherwise
        error('Unknown cfg.initial_mean: "%s"', cfg.initial_mean)
end


%%% ── SELECT initial function covariance (Q_f) ──────────────────────
%   a_f, sigma2_f : amplitude and lengthscale of the SE kernel
% Automatically builds a cell array for multi-state kernels.

a_f      = 1e1;    sigma2_f   = 1e1;
a_f_g    = 1e-1;   sigma2_f_g = 1e0;

if iscell(kf)
    Q_f = { @(X,V) a_f*squexp(X,V,sigma2_f), zero_kernel; ...
            zero_kernel, @(X,V) a_f_g*squexp(X,V,sigma2_f_g) };
else
    Q_f = @(X,V) a_f*squexp(X,V,sigma2_f);
end


%%% ── SELECT disturbance covariance (Q_w) ────────────────────────────
%   a_w, sigma2_w : amplitude and lengthscale of the SE kernel
% Automatically builds a cell array for multi-state kernels.

a_w      = 1e-2;   sigma2_w   = 1e0;
a_w_g    = 1e-2;   sigma2_w_g = 1e0;

if iscell(kf)
    Q_w = { @(X,V) a_w*squexp(X,V,sigma2_w), zero_kernel; ...
            zero_kernel, @(X,V) a_w_g*squexp(X,V,sigma2_w_g) };
else
    Q_w = @(X,V) a_w*squexp(X,V,sigma2_w);
end


%%% ── measurement noise covariance (Q_v) ─────────────────────────────
sigma2_v = 1e-5;
Q_v = @(X,V) sigma2_v * eq(X(:), V(:).');


%% Generate Ground Truth

switch cfg.system
    case 'Discrete_approximation'
        fprintf('True system: discrete approximation of the system functions\n\n')
        truth = DGP_simulation(cfg, kf, m, Q_f, Q_w, Q_v);
    case 'Heat_equation'
        fprintf('True system: 1-D heat equation\n\n')
        % Use kernel params if available; otherwise fall back to defaults.
        if ~exist('alpha','var'),   alpha   = 1; end
        if ~exist('delta_t','var'), delta_t = 1; end
        truth = DGP_heat_equation(cfg, alpha, delta_t, sigma2_v);
    case 'Wave_equation'
        fprintf('True system: 1-D wave equation\n\n')
        if ~exist('c_wave','var'),       c_wave       = 2;   end
        if ~exist('delta_t_wave','var'), delta_t_wave = 0.2; end
        truth = DGP_wave_equation(cfg, m, c_wave, delta_t_wave, sigma2_v);
    case 'Data'
        fprintf('True system: loaded dataset\n\n')
        [truth, cfg] = DGP_data(cfg);
    otherwise
        error('Unknown cfg.system: "%s"', cfg.system)
end

x      = truth.x;
y      = truth.y;
f_true = truth.f_true;


%% Fit Basis Functions to System Equations

fit = DGP_function_fitting(cfg, kf, m, Q_f, Q_w);


%% Run DGP Estimator

est = DGP_estimation(fit, x, y, Q_v);

% unpack for plotting convenience:
f_pred = est.f_pred;
f_upd  = est.f_upd;
c_pred = est.c_pred;
c_upd  = est.c_upd;

% estimation errors:
e_pred = f_true - f_pred(:,1:cfg.N);
e_upd  = f_true - f_upd;


%% Summary Plots

c_map = winter(cfg.N);

figure(100); clf
set(gcf, 'Units','normalized', 'Position',[0.05 0.05 0.9 0.9])

% (1) true function:
subplot(2,2,1)
plot_waterfall(cfg, x, y, f_true, [], c_map, false, ...
    'True function', '$f_{t}(x)$')

% (2) prediction step:
subplot(2,2,2)
plot_waterfall(cfg, x, y, f_pred(:,1:cfg.N), c_pred(:,:,1:cfg.N), c_map, ...
    cfg.plot_confidence_bounds, 'Estimate after prediction step', '$\hat{f}_{t|t-1}(x)$')

% (3) update step:
subplot(2,2,3)
plot_waterfall(cfg, x, y, f_upd, c_upd, c_map, ...
    cfg.plot_confidence_bounds, 'Estimate after update step', '$\hat{f}_{t|t}(x)$')

% (4) estimation error after update:
subplot(2,2,4)
plot_waterfall(cfg, [], [], e_upd, c_upd, c_map, ...
    cfg.plot_confidence_bounds, 'Estimation error after update step', '$\tilde{f}_{t|t}(x)$')


% MSE plot:
figure(1000)
set(gcf,'Position',[573 544 560 210])
plot(0:cfg.N-1, sqrt(sum(e_upd.^2, 1)), 'LineWidth',1, ...
     'DisplayName','$\Vert \tilde{f}_{t|t} \Vert_2$')
grid on; hold on
title('Estimation error 2-norm after update step','FontSize',15)
xlabel('$t$','FontSize',15)
ylabel('$\Vert \tilde{f}_{t|t} \Vert_2$','FontSize',15)
legend('Interpreter','latex')


%% ==================  Local Functions  ==================================


function plot_waterfall(cfg, x, y, f, c, c_map, show_cb, title_str, zlabel_str)
% plot_waterfall  3-D waterfall plot used by the summary figure.
%
%   cfg       configuration struct
%   x,y       measurement data (may be empty for error plots)
%   f         (N_test x N)  function values
%   c         (N_test x N_test x N) covariance (may be empty)
%   c_map     colormap matrix (N x 3)
%   show_cb   logical - whether to draw confidence bounds
%   title_str, zlabel_str  strings for title and z-label

    N = size(f, 2);
    for t = 1:N
        plot3(cfg.x_test, ones(cfg.N_test,1)*(t-1), f(:,t), ...
              'color', c_map(t,:), 'LineWidth',1.5)
        grid on; hold on

        if show_cb && ~isempty(c)
            cb = 2*sqrt(diag(c(:,:,t)));
            plot3(cfg.x_test, ones(cfg.N_test,1)*(t-1), f(:,t)-cb, ...
                  'color', c_map(t,:), 'LineWidth',0.5)
            plot3(cfg.x_test, ones(cfg.N_test,1)*(t-1), f(:,t)+cb, ...
                  'color', c_map(t,:), 'LineWidth',0.5)
        end

        if ~isempty(x) && ~isempty(y)
            plot3(x(:,t), ones(size(x,1),1)*(t-1), y(:,t), ...
                  'x', 'color', c_map(t,:), 'MarkerSize',15)
        end
    end
    title(title_str, 'FontSize',15)
    xlabel('$x$','FontSize',15);  ylabel('$t$','FontSize',15)
    zlabel(zlabel_str, 'FontSize',15)
    xlim([cfg.x_min cfg.x_max]);  ylim([0 N-1])
    view(-45, 30)
end
