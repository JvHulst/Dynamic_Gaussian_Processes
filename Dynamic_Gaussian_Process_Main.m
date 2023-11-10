%%% Dynamic_Gaussian_Processes %%%
% 
% 
% Jilles van Hulst
% 
% 17-02-2023
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
% 1) From a discretized approximation of the system functions (DGP_simulation.m)
% 2) From the solution to the 1D heat equation PDE (DGP_heat_equation.m)
% 3) From data (DGP_data.m)


close all
clear; clc


set(0,'defaultTextInterpreter','latex');


%% General Settings


% plotting settings:
plot_confidence_bounds = 1;     % if set to 1, the covariance estimates are plotted
plot_basis_functions = 0;       % if set to 1, the basis functions are plotted


% define evaluation range (gets overwritten if data is used):
N_test = 625;                    % number of points in function estimate evaluation
x_min = -10;                     % minimum input in function estimate evaluation
x_max = 10;                      % maximum input in function estimate evaluation


% number of observations at each timestep t (gets overwritten if data is used):
p = 5;


% duration of simulation (gets overwritten if data is used):
N = 11;


% number of bases:
M = 31;


% approximation basis choice:
% basis = 'RBF';
% basis = 'NRBF';      % normalized version of RBF
basis = 'Fourier';
% basis = 'Discrete';


% true system choice:
% system = 'Discrete_approximation';
system = 'Heat_equation';
% system = 'Data';


%% System functions


% evolution kernel:
alpha = 1;                  % thermal diffusivity
delta_t = 1;                % time-discretization sample time
kf = @(X,V) (  sqrt(delta_t*alpha)/2 * exp(-abs((X(:)-V(:).'))/sqrt(delta_t*alpha))  );         % time-discretized heat equation kernel
% sigma2_kf = 0.5;
% a_kf = 0.8;
% kf = @(X,V) a_kf*squexp(X,V,sigma2_kf)/sqrt(2*pi)/sigma2_kf;    % smoothening kernel, normalized to have area 1 before multiplication by a_kf
% kf = @(X,V) eq(X(:),V(:).')/(dx);                               % "integrator dynamics", kroneckerDelta


% initial function mean:
% m = @(X) zeros(size(X,1),1);                                    % zero mean
% m = @(X) ones(size(X,1),1);                                     % constant mean
m = @(X) 0.3*squexp(X,0,1);                                   % bell curve around origin
% m = @(X) -2*X.^2+8;                                             % parabola centered on origin
% m = @(X) double(abs(X(:))<=0.05)/0.1;                           % approximate impulse centered on origin


% initial function covariance:
a_f = 1e-2;
sigma2_f = 1e-1;
% Q_f = @(X,V) a_f*squexp(X,V,sigma2_f);                          % smooth initial condition variance
% Q_f = @(X,V) zeros(size(X,1),size(V,1));                        % no variance in i.c.
Q_f = @(X,V) sigma2_f*eq(X(:),V(:).');  	                      % kroneckerDelta: resembles white noise, spatially uncorrelated.


% disturbance covariance:
a_w = 1e-2;
sigma2_w = 1e-2;
% Q_w = @(X,V) a_w*squexp(X,V,sigma2_w);                          % smooth disturbances
% Q_w = @(X,V) zeros(size(X,1),size(V,1));                      % no disturbances
Q_w = @(X,V) sigma2_w*eq(X(:),V(:).');  	                    % kroneckerDelta: resembles white noise, spatially uncorrelated.


% measurement noise covariance:
sigma2_v = 1e-5;
Q_v = @(X,V) sigma2_v*eq(X(:),V(:).');                          % kroneckerDelta: resembles white noise, spatially uncorrelated.


%% Generate the 'true system' data


x_test = (x_min:(x_max-x_min)/(N_test-1):x_max).';

switch system
    case 'Discrete_approximation'
        fprintf('True system is a discrete approximation of the system functions \n\n')
        DGP_simulation;
    case 'Heat_equation'
        fprintf('True system is the 1-dimensional heat equation \n\n')
        DGP_heat_equation;
    case 'Data'
        fprintf('True system is a dataset \n\n')
        DGP_data;
end


%% Fit the system functions to the chosen set of basis functions


DGP_function_fitting;


%% Initialize Loop Variables


Psi_Gamma = NaN(M,p,N);

z_upd = NaN(M,N);
z_pred = NaN(M,N+1);

f_upd = NaN(N_test,N);
f_pred = NaN(N_test,N+1);

Psi_upd = NaN(M,M,N);
Psi_pred = NaN(M,M,N+1);

c_upd = NaN(N_test,N_test,N);
c_pred = NaN(N_test,N_test,N+1);

e_upd = NaN(N_test,N);
e_pred = NaN(N_test,N);


%% Initial Conditions


% initial condition of estimator:
z_pred(:,1) = z_bar;                    % prior mean estimate
% z_pred(:,1) = zeros(n_bases,1);         % prior mean estimate
Psi_pred(:,:,1) = Lambda_f;             % prior covariance estimate

c_pred(:,:,1) = U_test.'*Psi_pred(:,:,1)*U_test;
f_pred(:,1) = U_test.'*z_pred(:,1);


% plotting:
figure(1)
clf
plot(x_test,f_true(:,1),'--k','DisplayName','True function','LineWidth',1)
hold on; grid on
plot(x_test,f_pred(:,1),'r','DisplayName','Mean estimate','LineWidth',1)
if plot_confidence_bounds
    ucb = f_pred(:,1)+2*sqrt(diag(c_pred(:,:,1)));
    lcb = f_pred(:,1)-2*sqrt(diag(c_pred(:,:,1)));
    plot(x_test,ucb,'b:','HandleVisibility','off')
    plot(x_test,lcb,'b:','HandleVisibility','off')
    fill([x_test.', fliplr(x_test.')],[ucb.', fliplr(lcb.')],'cyan','FaceAlpha',0.3,'DisplayName','Estimated 95% conf.')
end
title('$t=0$')
xlabel('$x$')
ylabel('$f_{t}(x)$')
legend()


pause()


%% Simulation Loop


for t=1:N

    for i=1:M
        U_samp(i,:,t) = u{i}(x(:,t));      % evaluate the basis functions at the sample points
    end


    % DGP update:
    Psi_Gamma(:,:,t) = Psi_pred(:,:,t)*U_samp(:,:,t)/(U_samp(:,:,t).'*(Psi_pred(:,:,t))*U_samp(:,:,t)+Q_v(x(:,t),x(:,t)));
    if any(isnan(Psi_Gamma(:,:,t)))
        error('Psi_Gamma contains NaN')
    end
    z_upd(:,t) = z_pred(:,t)+Psi_Gamma(:,:,t)*(y(:,t)-U_samp(:,:,t).'*z_pred(:,t));
    Psi_upd(:,:,t) = Psi_pred(:,:,t)-Psi_Gamma(:,:,t)*U_samp(:,:,t).'*Psi_pred(:,:,t);


    % project onto bases:
    c_upd(:,:,t) = U_test.'*Psi_upd(:,:,t)*U_test;
    f_upd(:,t) = U_test.'*z_upd(:,t);

    e_pred(:,t) = f_true(:,t)-f_pred(:,t);
    e_upd(:,t) = f_true(:,t)-f_upd(:,t);


    % DGP prediction:
    z_pred(:,t+1) = Lambda*Lambda_U*z_upd(:,t);
    Psi_pred(:,:,t+1) = Lambda*Lambda_U*Psi_upd(:,:,t)*(Lambda*Lambda_U).' + Lambda_w;
    if any(isnan(Psi_pred(:,:,t+1)))
        error('Psi_pred contains NaN')
    end


    % project onto bases:
    c_pred(:,:,t+1) = U_test.'*Psi_pred(:,:,t+1)*U_test;
    f_pred(:,t+1) = U_test.'*z_pred(:,t+1);


    % plotting:
    figure(1)
    clf
    plot(x_test,f_true(:,t),'--k','DisplayName','True function','LineWidth',1)
    hold on; grid on
    plot(x(:,t),y(:,t),'rx','Markersize',15,'DisplayName','Observation')
    plot(x_test,f_upd(:,t),'r','DisplayName','Mean estimate','LineWidth',1)
    if plot_confidence_bounds
        ucb = f_upd(:,t)+2*sqrt(diag(c_upd(:,:,t)));
        lcb = f_upd(:,t)-2*sqrt(diag(c_upd(:,:,t)));
        plot(x_test,ucb,'b:','HandleVisibility','off')
        plot(x_test,lcb,'b:','HandleVisibility','off')
        fill([x_test.', fliplr(x_test.')],[ucb.', fliplr(lcb.')],'cyan','FaceAlpha',0.3,'DisplayName','Estimated 95% conf.')
    end
    title(sprintf('$t=%i$',t),'FontSize',15)
    xlabel('$x$','FontSize',15)
    ylabel('$f_{t}(x)$','FontSize',15)
    legend()

    pause(0.15)      % for the sake of visualization


end



%% Plotting


c = winter(N);


figure(100)
clf
set(gcf,"Units","normalized","Position",[0.05 0.05 0.9 0.9])

% true function:
subplot(2,2,1)
for t=1:N
    plot3(x_test,ones(size(x_test))*(t-1),f_true(:,t),'color',c(t,:),'LineWidth',1.5)
    grid on; hold on
    plot3(x(:,t),ones(size(x,1),1)*(t-1),y(:,t),'x','color',c(t,:),'MarkerSize',15)
end
title('True function','FontSize',15)
xlabel('$x$','FontSize',15)
ylabel('$t$','FontSize',15)
zlabel('$f_{t}(x)$','FontSize',15)
xlim([min(x_test) max(x_test)])
ylim([0 N-1])
view(-45,30)


% estimator function after prediction step:
subplot(2,2,2)
for t=1:N
    plot3(x_test,ones(size(x_test))*(t-1),f_pred(:,t),'color',c(t,:),'LineWidth',1.5)
    grid on; hold on

    if plot_confidence_bounds
        ucb = +2*sqrt(diag(c_pred(:,:,t)));
        lcb = -2*sqrt(diag(c_pred(:,:,t)));
        plot3(x_test,ones(size(x_test))*(t-1),f_pred(:,t)+lcb,'color',c(t,:),'LineWidth',0.5)
        plot3(x_test,ones(size(x_test))*(t-1),f_pred(:,t)+ucb,'color',c(t,:),'LineWidth',0.5)
    end

    plot3(x(:,t),ones(size(x,1),1)*(t-1),y(:,t),'x','color',c(t,:),'MarkerSize',15)
end
title('Estimate after prediction step','FontSize',15)
xlabel('$x$','FontSize',15)
ylabel('$t$','FontSize',15)
zlabel('$\hat{f}_{t|t-1}(x)$','FontSize',15)
xlim([min(x_test) max(x_test)])
ylim([0 N-1])
view(-45,30)


% estimator function after update step:
subplot(2,2,3)
for t=1:N
    plot3(x_test,ones(size(x_test))*(t-1),f_pred(:,t),'color',c(t,:),'LineWidth',1.5)
    grid on; hold on

    if plot_confidence_bounds
        ucb = +2*sqrt(diag(c_upd(:,:,t)));
        lcb = -2*sqrt(diag(c_upd(:,:,t)));
        plot3(x_test,ones(size(x_test))*(t-1),f_upd(:,t)+lcb,'color',c(t,:),'LineWidth',0.5)
        plot3(x_test,ones(size(x_test))*(t-1),f_upd(:,t)+ucb,'color',c(t,:),'LineWidth',0.5)
    end

    plot3(x(:,t),ones(size(x,1),1)*(t-1),y(:,t),'x','color',c(t,:),'MarkerSize',15)
end
title('Estimate after update step','FontSize',15)
xlabel('$x$','FontSize',15)
ylabel('$t$','FontSize',15)
zlabel('$\hat{f}_{t|t}(x)$','FontSize',15)
xlim([min(x_test) max(x_test)])
ylim([0 N-1])
view(-45,30)


% estimation error update step function:
subplot(2,2,4)
for t=1:N
    plot3(x_test,ones(size(x_test))*(t-1),e_upd(:,t),'color',c(t,:),'LineWidth',1.5)

    grid on; hold on
    
    if plot_confidence_bounds
        ucb = +2*sqrt(diag(c_upd(:,:,t)));
        lcb = -2*sqrt(diag(c_upd(:,:,t)));
    
        plot3(x_test,ones(size(x_test))*(t-1),lcb,'color',c(t,:),'LineWidth',0.5)
        plot3(x_test,ones(size(x_test))*(t-1),ucb,'color',c(t,:),'LineWidth',0.5)
    end
end
title('Estimation error after update step','FontSize',15)
xlabel('$x$','FontSize',15)
ylabel('$t$','FontSize',15)
zlabel('Estimation error','FontSize',15)
zlabel('$\tilde{f}_{t|t}(x)$','FontSize',15)
xlim([min(x_test) max(x_test)])
ylim([0 N-1])
view(-45,30)


% MSE of estimator after update step over timesteps:
figure(1000)
set(gcf,'Position',[573 544 560 210])
plot(0:N-1,sqrt(sum((e_upd).^2,1)),'LineWidth',1,'DisplayName','$\Vert \tilde{f}_{t|t} \Vert_2$')
grid on; hold on
title('Estimation error 2-norm after update step','FontSize',15)
xlabel('$t$','FontSize',15)
ylabel('$\Vert \tilde{f}_{t|t} \Vert_2$','FontSize',15)
legend('Interpreter','latex')


%% Functions


function K = squexp(U,V,sigma)
    % input U (N by n): 
    % input V (M by n):
    % input sigma (1 by 1): length scale. Assumed equal in every direction
    %
    % output K (N by M): matrix of squared exponentials

    N = size(U,1);
    M = size(V,1);

    K = NaN(N,M);

    for i=1:N
        for j=1:M
            diff = U(i,:)-V(j,:);
            K(i,j) = exp(-(diff/sigma^2*diff.')/2);
        end
    end

end

