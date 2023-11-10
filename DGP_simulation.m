%%% DGP_simulation %%%
% 
% 
% Jilles van Hulst
% 
% 10-11-2023
% 
% 
% Description:
% Generation of the 'true' function data for DGP estimator based on an
% approximation of the system functions with discrete bins.
% Generates the following variables:
% x: [p by N]                           measurement locations for every timestep
% y: [p by N]                           measurements for every timestep
%
% f_true: [N_test by N]                  true function evaluated at x_test for very timestep



%% Approximate the system functions using M_true discrete bins


% number of discrete bins used to fit the system functions:
M_true = 1001;


% create equally spaced set of points in function domain:
dx = (x_max-x_min)/(M_true-1);
x_fit = (x_min:dx:x_max).';


% discrete bin bounds:
bounds = x_min:(x_max*(1+1e-12)-x_min)/(M_true):x_max*(1+1e-12); % +1e-12 added to ensure bounds themselves fall in a base


% initialize basis function variables:
u_true = cell(M_true,1);
U_fit_true = NaN(M_true);
U_test_true = NaN(M_true,N_test);
U_samp_true = NaN(M_true,p);


% create and evaluate the basis functions:
for i=1:M_true
    u_true{i} = @(X) (double(bounds(i)<=X & X<bounds(i+1)));
    U_fit_true(i,:) = u_true{i}(x_fit);
    U_test_true(i,:) = u_true{i}(x_test);
end


% fitting of system (simple since orthonormal):
Lambda_U_true = U_fit_true.'*U_fit_true*dx;

Pinv_U_true = (U_fit_true.'*U_fit_true)\U_fit_true.';


% evolution matrix:
Lambda_true = Pinv_U_true*kf(x_fit,x_fit);



% initial conditions of the function:
z_bar_true = Pinv_U_true*m(x_fit);

Lambda_f_true = Pinv_U_true*Q_f(x_fit,x_fit);


% disturbance covariance matrices:
Lambda_w_true = Pinv_U_true*Q_w(x_fit,x_fit);


% make all covariance matrices symmetrical in case they are not:
Lambda_f_true = (Lambda_f_true.'+Lambda_f_true)/2;
Lambda_w_true = (Lambda_w_true.'+Lambda_w_true)/2;


if max(abs(eig(Lambda_true*Lambda_U_true)))<1
    fprintf('''True'' system is stable \n')
else
    fprintf('''True'' system is NOT stable \n')
end



%% Initialize Loop Variables


x = NaN(p,N);
v = NaN(p,N);
y = NaN(p,N);
z = NaN(M_true,N);
xi = NaN(M_true,N);
w = NaN(N_test,N);

f_true = NaN(N_test,N);
f_true_samp = NaN(p,N);


% initial condition of system at t=0:
z(:,1) = mvnrnd(z_bar_true,Lambda_f_true);



%% Simulate 'true' system

for t=1:N

    % measurement points:
    x(:,t) = unifrnd(x_min,x_max,p,1);        % random sample point in \mathcal{X}

    for i=1:M_true
        U_samp_true(i,:) = u_true{i}(x(:,t));      % evaluate the basis functions at the sample points
    end


    % evaluate true noiseless function:
    f_true(:,t) = U_test_true.'*z(:,t);            % at test points
    f_true_samp(:,t) = U_samp_true.'*z(:,t);       % at sample points


    % measure:
    v(:,t) = mvnrnd(zeros(p,1),Q_v(x(:,t),x(:,t)));                     % measurement noise
    y(:,t) = f_true_samp(:,t) + v(:,t);                                 % measurements


    % dynamics:
    xi(:,t) = mvnrnd(zeros(size(Lambda_w_true,1),1),Lambda_w_true);     % disturbance in the state variable z
    w(:,t) = U_test_true.'*xi(:,t);                                      % functional disturbance
    z(:,t+1) = Lambda_true*Lambda_U_true*z(:,t) + xi(:,t);


end


