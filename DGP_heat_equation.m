%%% DGP_heat_equation %%%
% 
% 
% Jilles van Hulst
% 
% 10-11-2023
% 
% 
% Description:
% Generation of the 'true' function data for DGP estimator based on a
% simulation of the heat equation PDE in 1 spatial dimension, which we
% sample at discrete time intervals.
% Assumes dirac in the origin as initial condition and that the domain of x
% is the real number line.



%% heat equation settings


alpha_HE = 1;          % thermal diffusivity

delta_t_HE = 1;        % sample time


% measurement noise covariance matrix (inherits sigma2_v from Main.m):       
% Sigma2_v = sigma2_v*eye(p);   
Sigma2_v = zeros(p);


%% Initialize Loop Variables


x = NaN(p,N);
v = NaN(p,N);
y = NaN(p,N);

f_true = NaN(N_test,N);
f_true_samp = NaN(p,N);



%% Simulate 'true' system


for t=1:N

    % measurement points:
    x(:,t) = unifrnd(x_min,x_max,p,1);        % random sample point in \mathcal{X}


    % evaluate true noiseless function:
    f_true(:,t) = heat_equation(x_test,alpha_HE,delta_t_HE*t);             % at test points
    f_true_samp(:,t) = heat_equation(x(:,t),alpha_HE,delta_t_HE*t);       % at sample points

    
    % measure:
    v(:,t) = mvnrnd(zeros(p,1),Sigma2_v);                               % measurement noise
    y(:,t) = f_true_samp(:,t) + v(:,t);                                 % measurements


end


%% Functions


function f = heat_equation(x,alpha,time)
    % input x:      (N by 1)
    % input alpha:  (1)
    % input time:   (1)
    %
    % output f:     (N by 1)
    
    f = 1/sqrt(4*pi*alpha*time) * exp(-(x.^2)/(4*alpha*time));

end