%%% Dynamic_Gaussian_Processes %%%
% 
% 
% Jilles van Hulst
% 
% 17-02-2023
% 
% 
% Description:
% Extension of Gaussian processes for evolving functions.
% OR
% Extension of Kalman filter for infinite-dimensional state (functions).
% This script considers an approximate solution using basis functions which
% allows for recursive and fast computation. The script automatically
% approximates user-defined functions.


%% Settings

close all
clear; clc

rng(1)

% plotting settings:
plot_confidence_bounds = 1;     % if set to 1, the covariance estimates are plotted

% define evaluation range:
Ntest = 625;
x_min = -1;
x_max = 1;

dx = (x_max-x_min)/(Ntest-1);

xtest = (x_min:dx:x_max).';

% number of observations at each timestep k:
Nsamp = 3;

% duration of simulation:
t_end = 7;



%% System functions

% evolution kernel:
kf_l = 0.04;
kf = @(X,V) squexp(X,V,kf_l)/sqrt(2*pi)/kf_l*0.70;   % normalized to have area 1. final multiplication ensures decay
% kf = @(X,V) sin(X(:)*pi)*sin(V(:)*pi).';
% kf = @(X,V) 2*X(:)*(2*(V(:)<=0).'-2*(V(:)>0).');
% kf = @(X,V) eq(X(:),V(:).')/(dx);                       % "integrator dynamics", kroneckerDelta

% initial function mean:
% m = @(X) zeros(size(X,1),1);                            % zero mean
% m = @(X) ones(size(X,1),1);                             % constant mean
% m = @(X) 10*squexp(X,0,0.02);                            % bell curve around origin
% m = @(X) -2*X.^2+8;                                     % parabola centered on origin
% m = @(X) eq(X(:),0)/(dx);                               % impulse centered on origin
m = @(X) double(abs(X(:))<=0.05)/0.1;                    % approximate impulse centered on origin

% initial function covariance:
sigma2_f = 0.3;
Q_f = @(X,V) sigma2_f*squexp(X,V,0.3);              % smooth initial condition variance
% Q_f = @(X,V) zeros(size(X,1),size(V,1));            % no variance in i.c.
% Q_f = @(X,V) sigma2_f*eq(X(:),V(:).');  	        % kroneckerDelta: resembles white noise, spatially uncorrelated.

% disturbance covariance:
sigma2_v = 0.05;
Q_v = @(X,V) sigma2_v*squexp(X,V,0.1);              % smooth disturbances
% Q_v = @(X,V) zeros(size(X,1),size(V,1));            % no disturbances
% Q_v = @(X,V) sigma2_v*eq(X(:),V(:).');  	        % kroneckerDelta: resembles white noise, spatially uncorrelated.

% measurement noise covariance:
sigma2_w = 0.1;
Q_w = @(X,V) sigma2_w*eq(X(:),V(:).');              % kroneckerDelta: resembles white noise, spatially uncorrelated.



%% Ground Truth Basis Functions (always discrete bases)

bounds = x_min:(x_max*(1+1e-12)-x_min)/(Ntest):x_max*(1+1e-12); % +1e-10 added to ensure bounds themselves fall in a base

u_true = cell(Ntest);
Utest_true = NaN(Ntest,Ntest);
Usamp_true = NaN(Ntest,Nsamp);

for i=1:Ntest
    u_true{i} = @(X) (double(bounds(i)<=X & X<bounds(i+1)));
    Utest_true(i,:) = u_true{i}(xtest);       % evaluate the basis functions at the test points
end


% fitting of system (simple since orthonormal):

Lambda_U_true = Utest_true.'*Utest_true*dx;

Pinv_U_true = (Utest_true.'*Utest_true)\Utest_true.';


% evolution matrix:

Lambda_true = Pinv_U_true*kf(xtest,xtest);
A_true = Lambda_true*Lambda_U_true;

if max(abs(eig(A_true)))<1
    fprintf('True system is stable \n')
else
    fprintf('True system is NOT stable \n')
end



% initial conditions of the function:

z_bar_true = Pinv_U_true*m(xtest);

Lambda_f_true = Pinv_U_true*Q_f(xtest,xtest);


% disturbance covariance matrices:
Lambda_V_true = Pinv_U_true*Q_v(xtest,xtest);


% make all covariance matrices symmetrical:
Lambda_f_true = (Lambda_f_true.'+Lambda_f_true)/2;
Lambda_V_true = (Lambda_V_true.'+Lambda_V_true)/2;



%% Approximation Basis Functions


% number of bases:
n_bases = 21;

% basis choice:
% basis = 'RBF';
% basis = 'NRBF';      % normalized version of RBF
basis = 'Fourier';
% basis = 'Discrete';

u = cell(n_bases);
Utest = NaN(n_bases,Ntest);
Usamp = NaN(n_bases,Nsamp,t_end);

switch basis
    case 'RBF'
        orthogonal = false;
        if isequal(n_bases,1)
            db = (x_max-x_min)/2;
            center = (x_max+x_min)/2;
        else
            db = (x_max-x_min)/(n_bases-1);
            center = x_min:db:x_max;
        end

        l = db*0.8;
        
        for i=1:n_bases
            u{i} = @(X) (squexp(X,center(i),l));
            Utest(i,:) = u{i}(xtest);       % evaluate the basis functions at the test points
        end
    case 'NRBF'
        orthogonal = false;
        if isequal(n_bases,1)
            db = (x_max-x_min)/2;
            center = (x_max+x_min)/2;
        else
            db = (x_max-x_min)/(n_bases-1);
            center = x_min:db:x_max;
        end

        l = db*0.6;
        
        Unorm = @(X) sum(squexp(X,center.',l),2);
        
        for i=1:n_bases
            u{i} = @(X) (squexp(X,center(i),l)./Unorm(X));
            Utest(i,:) = u{i}(xtest);       % evaluate the basis functions at the test points
        end

    case 'Fourier'
        orthogonal = true;
        n = 0;
        period = x_max-x_min;
        
        for i=1:n_bases
            if rem(i,2)==1
                u{i} = @(X) (Fourier_cos(X,n,period));
                n = n+1;
            else
                u{i} = @(X) (Fourier_sin(X,n,period));
            end
            Utest(i,:) = u{i}(xtest);       % evaluate the basis functions at the test points
        end
    case 'Discrete'
        orthogonal = true;
        db = (x_max+1e-12-x_min)/(n_bases);     % +1e-12 added to ensure every point in xtest is in only 1 base.
        bounds = x_min:db:x_max+1e-12;

        for i=1:n_bases
            u{i} = @(X) (double(bounds(i)<=X & X<bounds(i+1))/sqrt(db));
            Utest(i,:) = u{i}(xtest);       % evaluate the basis functions at the test points
        end
end

% plot the basis functions:
figure(1)
clf
hold on; grid on
plot(xtest,Utest)
xlabel('$x$')
ylabel('$u_i(x)$')



%% System projection


tic


U_mat = NaN(Ntest,Ntest,n_bases^2);
Lambda_U = NaN(n_bases,n_bases);

for i=1:n_bases
    for j=1:n_bases
        U_mat(:,:,(i-1)*n_bases+j) = Utest(i,:).'*Utest(j,:);
        Lambda_U(i,j) = sum(Utest(i,:).*Utest(j,:))*dx;
    end
end

U_mat = reshape(U_mat,[Ntest^2,n_bases^2]);


if orthogonal
% fitting of system (simplified since orthonormal):

%TODO gives a slightly different answer
    Pinv_U = diag(1./sum(U_mat.^2,1))*(U_mat.');
%     Pinv_U = (U_mat.'*U_mat)\(U_mat.');

else
    Pinv_U = (U_mat.'*U_mat)\(U_mat.');

end


% evolution matrix:
Kf = reshape(kf(xtest,xtest).',[Ntest^2,1]);

Lambda = reshape(Pinv_U*Kf,[n_bases,n_bases]);
A = Lambda*Lambda_U;


% initial conditions of the function:
QQf = reshape(Q_f(xtest,xtest),[Ntest^2,1]);

Lambda_f = reshape(Pinv_U*QQf,[n_bases,n_bases]);

z_bar = (Utest*Utest.')\Utest*m(xtest);


% disturbance covariance matrices:
QQv = reshape(Q_v(xtest,xtest),[Ntest^2,1]);

Lambda_V = reshape(Pinv_U*QQv,[n_bases,n_bases]);


toc


if max(abs(eig(A)))<1
    fprintf('Approximation is stable \n')
else
    fprintf('Approximation is NOT stable \n')
end


% make all covariance matrices symmetrical:
Lambda_f = (Lambda_f.'+Lambda_f)/2;
Lambda_V = (Lambda_V.'+Lambda_V)/2;

%% Initialize Loop Variables

% 'true' system:

x = NaN(Nsamp,t_end);
w = NaN(Nsamp,t_end);
y = NaN(Nsamp,t_end);
z = NaN(Ntest,t_end);
xi = NaN(Ntest,t_end);
v = NaN(Ntest,t_end);

f = NaN(Ntest,t_end);
f_samp = NaN(Nsamp,t_end);

% estimator:

Psi_Gamma = NaN(n_bases,Nsamp,t_end);

z_upd = NaN(n_bases,t_end);
z_pred = NaN(n_bases,t_end+1);

f_upd = NaN(Ntest,t_end);
f_pred = NaN(Ntest,t_end+1);

Psi_upd = NaN(n_bases,n_bases,t_end);
Psi_pred = NaN(n_bases,n_bases,t_end+1);

c_upd = NaN(Ntest,Ntest,t_end);
c_pred = NaN(Ntest,Ntest,t_end+1);

e_upd = NaN(Ntest,t_end);
e_pred = NaN(Ntest,t_end);



%% Initial Conditions

% initial condition of system at k=1:
z(:,1) = mvnrnd(z_bar_true,Lambda_f_true);

% initial condition of estimator:
z_pred(:,1) = z_bar;                    % prior mean estimate
% z_pred(:,1) = zeros(n_bases,1);         % prior mean estimate
Psi_pred(:,:,1) = Lambda_f;             % prior covariance estimate

c_pred(:,:,1) = Utest.'*Psi_pred(:,:,1)*Utest;
f_pred(:,1) = Utest.'*z_pred(:,1);

% plotting:
figure(2)
clf
plot(xtest,Utest_true.'*z(:,1),'--k','DisplayName','True function','LineWidth',1)
hold on; grid on
plot(xtest,f_pred(:,1),'r','DisplayName','Mean estimate','LineWidth',1)
if plot_confidence_bounds
ucb = f_pred(:,1)+2*sqrt(diag(c_pred(:,:,1)));
lcb = f_pred(:,1)-2*sqrt(diag(c_pred(:,:,1)));
plot(xtest,ucb,'b:','HandleVisibility','off')
plot(xtest,lcb,'b:','HandleVisibility','off')
fill([xtest.', fliplr(xtest.')],[ucb.', fliplr(lcb.')],'cyan','FaceAlpha',0.3,'DisplayName','Estimated 95% conf.')
end
title('$t=0$')
xlabel('$x$')
ylabel('$f_{t}(x)$')
legend()

pause()



%% Simulation Loop

for t=1:t_end

    % measurement points:
    x(:,t) = unifrnd(x_min,x_max,Nsamp,1);        % random sample point in \mathcal{X}

    for i=1:Ntest
        Usamp_true(i,:) = u_true{i}(x(:,t));      % evaluate the basis functions at the sample points
    end

    for i=1:n_bases
        Usamp(i,:,t) = u{i}(x(:,t));      % evaluate the basis functions at the sample points
    end

    % evaluate true noiseless function:
    f(:,t) = Utest_true.'*z(:,t);            % at test points
    f_samp(:,t) = Usamp_true.'*z(:,t);       % at sample points

    % measure:
    w(:,t) = mvnrnd(zeros(Nsamp,1),Q_w(x(:,t),x(:,t)));
    y(:,t) = f_samp(:,t) + w(:,t);


    % DGP update:
    Psi_Gamma(:,:,t) = Psi_pred(:,:,t)*Usamp(:,:,t)/(Usamp(:,:,t).'*(Psi_pred(:,:,t))*Usamp(:,:,t)+Q_w(x(:,t),x(:,t)));
    if any(isnan(Psi_Gamma(:,:,t)))
        error('Psi_Gamma contains NaN')
    end
    z_upd(:,t) = z_pred(:,t)+Psi_Gamma(:,:,t)*(y(:,t)-Usamp(:,:,t).'*z_pred(:,t));
    Psi_upd(:,:,t) = Psi_pred(:,:,t)-Psi_Gamma(:,:,t)*Usamp(:,:,t).'*Psi_pred(:,:,t);

    % project onto bases:
    c_upd(:,:,t) = Utest.'*Psi_upd(:,:,t)*Utest;
    f_upd(:,t) = Utest.'*z_upd(:,t);

    e_pred(:,t) = f(:,t)-f_pred(:,t);
    e_upd(:,t) = f(:,t)-f_upd(:,t);

    % dynamics:
    xi(:,t) = mvnrnd(zeros(size(Lambda_V_true,1),1),Lambda_V_true);
    v(:,t) = Utest_true.'*xi(:,t);
    z(:,t+1) = Lambda_true*Lambda_U_true*z(:,t) + xi(:,t);


    % DGP prediction:
    z_pred(:,t+1) = Lambda*Lambda_U*z_upd(:,t);
    Psi_pred(:,:,t+1) = Lambda*Lambda_U*Psi_upd(:,:,t)*(Lambda*Lambda_U).' + Lambda_V;
    if any(isnan(Psi_pred(:,:,t+1)))
        error('Psi_pred contains NaN')
    end

    % project onto bases:
    c_pred(:,:,t+1) = Utest.'*Psi_pred(:,:,t+1)*Utest;
    f_pred(:,t+1) = Utest.'*z_pred(:,t+1);


    % plotting:
    figure(2)
    clf
    plot(xtest,f(:,t),'--k','DisplayName','True function','LineWidth',1)
    hold on; grid on
    plot(x(:,t),y(:,t),'rx','Markersize',15,'DisplayName','Observation')
    plot(xtest,f_upd(:,t),'r','DisplayName','Mean estimate','LineWidth',1)
    if plot_confidence_bounds
    ucb = f_upd(:,t)+2*sqrt(diag(c_upd(:,:,t)));
    lcb = f_upd(:,t)-2*sqrt(diag(c_upd(:,:,t)));
    plot(xtest,ucb,'b:','HandleVisibility','off')
    plot(xtest,lcb,'b:','HandleVisibility','off')
    fill([xtest.', fliplr(xtest.')],[ucb.', fliplr(lcb.')],'cyan','FaceAlpha',0.3,'DisplayName','Estimated 95% conf.')
    end
    title(sprintf('$t=%i$',t),'FontSize',15)
    xlabel('$x$','FontSize',15)
    ylabel('$f_{t}(x)$','FontSize',15)
    legend()

    pause(0.15)      % for the sake of visualization

end



%% Plotting


figure(10)
clf

subplot(2,2,1)
plot(0:t_end,z.')
grid on; hold on
xlabel('$t$','Interpreter','latex','FontSize',15)
ylabel('$z_t$','Interpreter','latex','FontSize',15)
xlim([0 t])
% xticks(0:k)

subplot(2,2,2)
plot(1:t_end,y.')
grid on; hold on
xlabel('$t$','Interpreter','latex','FontSize',15)
ylabel('$y_t$','Interpreter','latex','FontSize',15)
xlim([0 t_end-1])
% xticks(0:k-1)

subplot(2,2,3)
plot(0:t_end-1,z_upd.')
grid on; hold on
xlabel('$t$','Interpreter','latex','FontSize',15)
ylabel('$\hat{z}_{t|t}$','Interpreter','latex','FontSize',15)
xlim([0 t-1])
% xticks(0:k-1)

subplot(2,2,4)
plot(0:t_end,z_pred.')
grid on; hold on
xlabel('$t$','Interpreter','latex','FontSize',15)
ylabel('$\hat{z}_{t|t-1}$','Interpreter','latex','FontSize',15)
xlim([0 t_end])
% xticks(0:k)


t_end_plot = min([12,t_end]);

c = winter(t_end_plot);
% c = flipud(c);


figure(100)
clf
set(gcf,'Position',[573 544 560 300])
for t=1:t_end_plot
    plot3(xtest,ones(size(xtest))*(t-1),f(:,t),'color',c(t,:),'LineWidth',1.5)
%     surf([xtest xtest],ones(size(xtest))*(t-1)+[-0.25 0.25],[f(:,t) f(:,t)],[f(:,t) f(:,t)],'EdgeColor','none');
    grid on; hold on
    plot3(x(:,t),ones(size(x,1),1)*(t-1),y(:,t),'x','color',c(t,:),'MarkerSize',15)
end
xlabel('$x$','FontSize',15)
ylabel('$t$','FontSize',15)
zlabel('$f_{t}(x)$','FontSize',15)
xlim([min(xtest) max(xtest)])
ylim([0 t_end_plot-1])
% colormap jet
zlim([-3 10.5])
% view(3)
view(-45,30)


figure(200)
clf
for t=1:t_end_plot
    plot3(xtest,ones(size(xtest))*(t-1),f_pred(:,t),'color',c(t,:),'LineWidth',1.5)
    grid on; hold on
    plot3(x(:,t),ones(size(x,1),1)*(t-1),y(:,t),'x','color',c(t,:),'MarkerSize',15)
end
% plot3(x(:,1:t_end_plot),ones(size(x,1),1)*(0:t_end_plot-1),y(:,1:t_end_plot),'rx','MarkerSize',15)
xlabel('$x$','FontSize',15)
ylabel('$t$','FontSize',15)
zlabel('$\hat{f}_{t|t-1}(x)$','FontSize',15)
xlim([min(xtest) max(xtest)])
ylim([0 t_end_plot-1])
% view(2)


figure(300)
clf
for t=1:t_end_plot
    plot3(xtest,ones(size(xtest))*(t-1),f_upd(:,t),'color',c(t,:),'LineWidth',1.5)
%     surf([xtest xtest],ones(size(xtest))*(t-1)+[-0.25 0.25],[f_upd(:,t) f_upd(:,t)],[f_upd(:,t) f_upd(:,t)],'EdgeColor','none');
    grid on; hold on
    plot3(x(:,t),ones(size(x,1),1)*(t-1),y(:,t),'x','color',c(t,:),'MarkerSize',15)
end
% plot3(x(:,1:t_end_plot),ones(size(x,1),1)*(0:t_end_plot-1),y(:,1:t_end_plot),'rx','MarkerSize',15)
xlabel('$x$','FontSize',15)
ylabel('$t$','FontSize',15)
zlabel('$\hat{f}_{t|t}(x)$','FontSize',15)
xlim([min(xtest) max(xtest)])
ylim([0 t_end_plot-1])
% view(2)

figure(301)
clf
set(gcf,'Position',[573 544 560 300])
for t=1:t_end_plot
    plot3(xtest,ones(size(xtest))*(t-1),f_upd(:,t),'color',c(t,:),'LineWidth',1.5)
    grid on; hold on

    ucb = +2*sqrt(diag(c_upd(:,:,t)));
    lcb = -2*sqrt(diag(c_upd(:,:,t)));

    plot3(xtest,ones(size(xtest))*(t-1),f_upd(:,t)+lcb,'color',c(t,:),'LineWidth',0.5)
    plot3(xtest,ones(size(xtest))*(t-1),f_upd(:,t)+ucb,'color',c(t,:),'LineWidth',0.5)
    plot3(x(:,t),ones(size(x,1),1)*(t-1),y(:,t),'x','color',c(t,:),'MarkerSize',15)
end
% plot3(x(:,1:t_end_plot),ones(size(x,1),1)*(0:t_end_plot-1),y(:,1:t_end_plot),'rx','MarkerSize',15)
xlabel('$x$','FontSize',15)
ylabel('$t$','FontSize',15)
zlabel('$\hat{f}_{t|t}(x)$','FontSize',15)
xlim([min(xtest) max(xtest)])
ylim([0 t_end_plot-1])
zlim([-3 10.5])
% view(3)
view(-45,30)


figure(400)
clf
for t=1:t_end_plot
    plot3(xtest,ones(size(xtest))*(t-1),e_upd(:,t),'color',c(t,:),'LineWidth',1.5)
%     surf([xtest xtest],ones(size(xtest))*(t-1)+[-0.25 0.25],[e_upd(:,t) e_upd(:,t)],[e_upd(:,t) e_upd(:,t)],'EdgeColor','none');
    grid on; hold on
end
xlabel('$x$','FontSize',15)
ylabel('$t$','FontSize',15)
zlabel('Estimation error','FontSize',15)
zlabel('$\tilde{f}_{t|t}(x)$','FontSize',15)
xlim([min(xtest) max(xtest)])
ylim([0 t_end_plot-1])
% view(2)

% figure(401)
% clf
% surf(x_grid.',k_grid.',e_upd,'EdgeColor','none')
% xlabel('$x$')
% ylabel('$t$')
% zlabel('Estimation error')
% zlabel('$f_{t}(x) - \hat{f}_{t|t}(x)$')
% colormap pink


figure(402)
clf
for t=1:t_end_plot
    plot3(xtest,ones(size(xtest))*(t-1),e_upd(:,t),'color',c(t,:),'LineWidth',1.5)

    grid on; hold on

    ucb = +2*sqrt(diag(c_upd(:,:,t)));
    lcb = -2*sqrt(diag(c_upd(:,:,t)));

    plot3(xtest,ones(size(xtest))*(t-1),lcb,'color',c(t,:),'LineWidth',0.5)
    plot3(xtest,ones(size(xtest))*(t-1),ucb,'color',c(t,:),'LineWidth',0.5)
end
xlabel('$x$','FontSize',15)
ylabel('$t$','FontSize',15)
zlabel('Estimation error','FontSize',15)
zlabel('$\tilde{f}_{t|t}(x)$','FontSize',15)
xlim([min(xtest) max(xtest)])
ylim([0 t_end_plot-1])
% view(2)


figure(500)
% clf
set(gcf,'Position',[573 544 560 210])
plot(0:t_end-1,sqrt(sum((e_pred).^2,1)),'LineWidth',1,'DisplayName','$\Vert \tilde{f}_{t|t-1} \Vert_2$')
grid on; hold on
% plot(0:t_end-1,sqrt(sum((e_upd).^2,1)),'LineWidth',1,'DisplayName','$\Vert \tilde{f}_{t|t} \Vert_2$')
for t=1:t_end
    two_norm_expected(t) = sqrt(eigs(Utest.'*Psi_pred(:,:,t)*Utest,1));
%     Psi_inf(:,:,k) = idare(A.',Usamp(:,:,k),Lambda_V,Theta_w(x(:,k),x(:,k)));
%     two_norm_infinite(k) = sqrt(eigs(Utest.'*Psi_inf(:,:,k)*Utest,1));
end
% plot(0:t_end-1,two_norm_expected,'k:','LineWidth',1,'DisplayName','Expected $\Vert e_t \Vert_2$')
% plot(0:k_end-1,two_norm_infinite,'r:','LineWidth',1)
xlabel('$t$','FontSize',15)
ylabel('$\Vert \tilde{f}_{t|t-1} \Vert_2$','FontSize',15)
legend('Interpreter','latex')


%% Functions

function K = squexp(U,V,l)
    % input U (N by n): 
    % input V (M by n):
    % input l (1 by 1):  length scale
    % output K (N by M): matrix of squared exponentials

    N = size(U,1);
    M = size(V,1);

    K = NaN(N,M);

    for i=1:N
        for j=1:M
            diff = U(i,:)-V(j,:);
            K(i,j) = exp(-(diff/l^2*diff.')/2);
        end
    end
%     diff = U(:)-V(:).';
%     K = exp(-(diff.^2)/(2*l^2));           % not quite right. Mind multi-dimensionality and evaluation over multiple datapoints

end

function U = Fourier_sin(X,n,period)
    % input X: (N by 1)
    % output U: (N by 1)
    
    U = sin(2*pi*n*X(:)/period);

end

function U = Fourier_cos(X,n,period)
    % input X: (N by 1)
    % output U: (N by 1)
    
    U = cos(2*pi*n*X(:)/period);

end
