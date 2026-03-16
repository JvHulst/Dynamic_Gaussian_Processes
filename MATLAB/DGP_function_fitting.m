function fit = DGP_function_fitting(cfg, kf, m, Q_f, Q_w)
%%% DGP_function_fitting %%%
%
% Jilles van Hulst
%
% 10-11-2023  (refactored 2026)
%
%
% Description:
%   Fits the approximate DGP framework to the system equations defined in
%   the calling script. The basis functions are fitted to the system
%   functions evaluated at a grid of inputs using linear least squares.
%
% Inputs:
%   cfg   struct with fields:
%           .basis  ('RBF','NRBF','Fourier','Discrete')
%           .M      number of basis functions
%           .x_min, .x_max   domain bounds
%           .N_test           number of evaluation test points
%           .x_test           (N_test x 1) test input grid
%           .plot_basis_functions  (logical)
%
%   kf    evolution kernel - function handle for single-state, or
%         {n_states x n_states} cell array of function handles for
%         multi-state.  The number of states is inferred automatically.
%
%   m     initial function mean - function handle  @(X) -> (N x 1).
%         For multi-state systems, m specifies the first state only;
%         additional states are initialised at zero.
%
%   Q_f   initial function covariance - function handle or cell array.
%   Q_w   disturbance covariance     - function handle or cell array.
%
% Outputs:
%   fit   struct with fields:
%           .Lambda    (n_s x n_s)   fitted evolution matrix
%           .Lambda_U  (n_s x n_s)   basis inner-product / Gram matrix
%           .Lambda_f  (n_s x n_s)   fitted initial covariance
%           .Lambda_w  (n_s x n_s)   fitted disturbance covariance
%           .z_bar     (n_s x 1)     fitted initial mean
%           .u         {M x 1}       cell array of basis function handles
%           .U_test    (n_s x N_test) basis functions at test points
%                                     (zero-padded for unobserved states)
%           .n_states  number of function-valued states
%           .n_s       total state dimension  =  n_states * M


%% Unpack configuration

M      = cfg.M;
x_min  = cfg.x_min;
x_max  = cfg.x_max;
N_test = cfg.N_test;
x_test = cfg.x_test;


%% Normalise to cell-array format

% This lets a single code path handle both single-state and multi-state.
if ~iscell(kf)
    kf  = {kf};
    Q_f = {Q_f};
    Q_w = {Q_w};
end
n_states = size(kf, 1);
n_s = n_states * M;


%% Approximation Basis Functions

N_fit  = N_test;
dx_fit = (x_max - x_min) / (N_fit - 1);
x_fit  = (x_min : dx_fit : x_max).';

u      = cell(M, 1);
U_fit  = NaN(M, N_fit);
U_test = NaN(M, N_test);

switch cfg.basis
    case 'RBF'
        orthogonal = false;
        if M == 1
            db = (x_max - x_min) / 2;
            center = (x_max + x_min) / 2;
        else
            db = (x_max - x_min) / (M - 1);
            center = x_min : db : x_max;
        end
        l = db * 0.8;
        for i = 1:M
            u{i} = @(X) squexp(X, center(i), l);
        end

    case 'NRBF'
        orthogonal = false;
        if M == 1
            db = (x_max - x_min) / 2;
            center = (x_max + x_min) / 2;
        else
            db = (x_max - x_min) / (M - 1);
            center = x_min : db : x_max;
        end
        l = db * 0.6;
        Unorm = @(X) sum(squexp(X, center.', l), 2);
        for i = 1:M
            u{i} = @(X) squexp(X, center(i), l) ./ Unorm(X);
        end

    case 'Fourier'
        orthogonal = true;
        n = 0;
        period = x_max - x_min;
        for i = 1:M
            if rem(i, 2) == 1
                u{i} = @(X) Fourier_cos(X, n, period) * sqrt(2/period);
                n = n + 1;
            else
                u{i} = @(X) Fourier_sin(X, n, period) * sqrt(2/period);
            end
        end

    case 'Discrete'
        orthogonal = true;
        db = (x_max + 1e-12 - x_min) / M;
        bounds = x_min : db : x_max + 1e-12;
        for i = 1:M
            u{i} = @(X) double(bounds(i) <= X & X < bounds(i+1)) / db;
        end
end

% evaluate basis functions at fitting and test points:
for i = 1:M
    U_fit(i,:)  = u{i}(x_fit);
    U_test(i,:) = u{i}(x_test);
end

% (optional) plot:
if cfg.plot_basis_functions
    figure(10); clf; hold on; grid on
    plot(x_test, U_test)
    xlabel('$x$'); ylabel('$u_i(x)$')
end


%% System projection

tic

% basis inner-product matrix (M x M):
U_mat    = NaN(N_fit, N_fit, M^2);
Lambda_U = NaN(M, M);

for i = 1:M
    for j = 1:M
        U_mat(:,:,(i-1)*M+j) = U_fit(i,:).' * U_fit(j,:);
        Lambda_U(i,j) = sum(U_fit(i,:) .* U_fit(j,:)) * dx_fit;
    end
end

U_mat = reshape(U_mat, [N_fit^2, M^2]);

if orthogonal
    Pinv_U = diag(1 ./ sum(U_mat.^2, 1)) * U_mat.';
else
    Pinv_U = (U_mat.' * U_mat) \ U_mat.';
end

% allocate block matrices:
Lambda   = NaN(n_s, n_s);
Lambda_f = NaN(n_s, n_s);
Lambda_w = NaN(n_s, n_s);
z_bar    = NaN(n_s, 1);

for ii = 1:n_states
    % initial mean: first state uses m(), remaining states start at zero
    if ii == 1
        z_bar(1:M) = (U_fit * U_fit.') \ U_fit * m(x_fit);
    else
        z_bar((ii-1)*M+1 : ii*M) = zeros(M, 1);
    end

    for jj = 1:n_states
        rows = (ii-1)*M+1 : ii*M;
        cols = (jj-1)*M+1 : jj*M;

        Kf_ij = reshape(kf{ii,jj}(x_fit, x_fit).', [N_fit^2, 1]);
        Lambda(rows, cols) = reshape(Pinv_U * Kf_ij, [M, M]);

        Qf_ij = reshape(Q_f{ii,jj}(x_fit, x_fit), [N_fit^2, 1]);
        Lambda_f(rows, cols) = reshape(Pinv_U * Qf_ij, [M, M]);

        Qw_ij = reshape(Q_w{ii,jj}(x_fit, x_fit), [N_fit^2, 1]);
        Lambda_w(rows, cols) = reshape(Pinv_U * Qw_ij, [M, M]);
    end
end

% extend Lambda_U to block-diagonal:
Lambda_U = kron(eye(n_states), Lambda_U);

% enforce symmetry of covariance matrices:
Lambda_f = (Lambda_f.' + Lambda_f) / 2;
Lambda_w = (Lambda_w.' + Lambda_w) / 2;

toc

% stability check:
eig_max = max(abs(eig(Lambda * Lambda_U)));
if eig_max < 1
    fprintf('Approximation is stable (max |eig| = %.4f)\n', eig_max)
elseif abs(eig_max - 1) < 1e-6
    fprintf('Approximation is marginally stable (max |eig| = %.6f)\n', eig_max)
else
    fprintf('Approximation is NOT stable (max |eig| = %.4f)\n', eig_max)
end


%% Extend U_test for multi-state

% Only the first state is observed / plotted.  Pad with zeros so that
%   U_test.' * z   extracts only the f-component.
if n_states > 1
    U_test = [U_test; zeros((n_states-1)*M, N_test)];
end


%% Pack outputs

fit.Lambda   = Lambda;
fit.Lambda_U = Lambda_U;
fit.Lambda_f = Lambda_f;
fit.Lambda_w = Lambda_w;
fit.z_bar    = z_bar;
fit.u        = u;
fit.U_test   = U_test;
fit.n_states = n_states;
fit.n_s      = n_s;

end


%% ==================  Local Functions  ==================================

function U = Fourier_sin(X, n, period)
    U = sin(2*pi*n*X(:) / period);
end

function U = Fourier_cos(X, n, period)
    U = cos(2*pi*n*X(:) / period);
    if n == 0
        U = U ./ sqrt(2);
    end
end
