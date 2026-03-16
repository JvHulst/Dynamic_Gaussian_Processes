function est = DGP_estimation(fit, x, y, Q_v)
%%% DGP_estimation %%%
%
% Jilles van Hulst
%
% 2026
%
%
% Description:
%   Runs the Dynamic Gaussian Process estimator (Kalman-filter recursion
%   on the basis-function coefficients).  This implements the update and
%   prediction equations from
%
%       "Estimation of Dynamic Gaussian Processes"
%       (van Hulst, van Zuijlen, Antunes, Heemels)
%
%   given a fitted approximation basis (from DGP_function_fitting) and a
%   set of noisy measurements.
%
% Inputs:
%   fit   struct returned by DGP_function_fitting, with fields:
%           .Lambda, .Lambda_U, .Lambda_f, .Lambda_w   system matrices
%           .z_bar          (n_s x 1)    initial mean
%           .u              {M x 1}      basis function handles
%           .U_test         (n_s x N_test) basis at test points
%           .n_states, .n_s
%
%   x     (p x N)   measurement locations for each timestep
%   y     (p x N)   noisy measurements    for each timestep
%   Q_v   function handle @(X,V) -> (p x p)  measurement noise covariance
%
% Outputs:
%   est   struct with fields:
%           .z_pred    (n_s x N+1)              predicted state mean
%           .z_upd     (n_s x N)                updated  state mean
%           .Psi_pred  (n_s x n_s x N+1)       predicted state covariance
%           .Psi_upd   (n_s x n_s x N)         updated  state covariance
%           .f_pred    (N_test x N+1)           predicted function mean
%           .f_upd     (N_test x N)             updated  function mean
%           .c_pred    (N_test x N_test x N+1)  predicted function covariance
%           .c_upd     (N_test x N_test x N)    updated  function covariance


%% Unpack

Lambda   = fit.Lambda;
Lambda_U = fit.Lambda_U;
Lambda_f = fit.Lambda_f;
Lambda_w = fit.Lambda_w;
z_bar    = fit.z_bar;
u        = fit.u;
U_test   = fit.U_test;
n_s      = fit.n_s;

M      = numel(u);           % number of basis functions per state
[p, N] = size(y);
N_test = size(U_test, 2);

A = Lambda * Lambda_U;       % state transition matrix (precompute)


%% Allocate

z_pred   = NaN(n_s, N+1);
z_upd    = NaN(n_s, N);

Psi_pred = NaN(n_s, n_s, N+1);
Psi_upd  = NaN(n_s, n_s, N);

f_pred   = NaN(N_test, N+1);
f_upd    = NaN(N_test, N);

c_pred   = NaN(N_test, N_test, N+1);
c_upd    = NaN(N_test, N_test, N);


%% Initial condition

z_pred(:,1)     = z_bar;
Psi_pred(:,:,1) = Lambda_f;

f_pred(:,1)   = U_test.' * z_pred(:,1);
c_pred(:,:,1) = U_test.' * Psi_pred(:,:,1) * U_test;


%% Estimation loop

for t = 1:N

    % --- evaluate basis functions at measurement locations ---
    U_samp = zeros(n_s, p);
    for i = 1:M
        U_samp(i,:) = u{i}(x(:,t));
    end
    % (rows M+1:n_s remain zero for unobserved states)

    % --- DGP update (measurement incorporation) ---
    Psi_Gamma = Psi_pred(:,:,t) * U_samp ...
        / (U_samp.' * Psi_pred(:,:,t) * U_samp + Q_v(x(:,t), x(:,t)));

    z_upd(:,t)     = z_pred(:,t) + Psi_Gamma * (y(:,t) - U_samp.' * z_pred(:,t));
    Psi_upd(:,:,t) = Psi_pred(:,:,t) - Psi_Gamma * U_samp.' * Psi_pred(:,:,t);

    f_upd(:,t)   = U_test.' * z_upd(:,t);
    c_upd(:,:,t) = U_test.' * Psi_upd(:,:,t) * U_test;

    % --- DGP prediction (time propagation) ---
    z_pred(:,t+1)     = A * z_upd(:,t);
    Psi_pred(:,:,t+1) = A * Psi_upd(:,:,t) * A.' + Lambda_w;

    f_pred(:,t+1)   = U_test.' * z_pred(:,t+1);
    c_pred(:,:,t+1) = U_test.' * Psi_pred(:,:,t+1) * U_test;

end


%% Pack outputs

est.z_pred   = z_pred;
est.z_upd    = z_upd;
est.Psi_pred = Psi_pred;
est.Psi_upd  = Psi_upd;
est.f_pred   = f_pred;
est.f_upd    = f_upd;
est.c_pred   = c_pred;
est.c_upd    = c_upd;

end
