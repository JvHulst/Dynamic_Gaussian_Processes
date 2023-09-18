%%% Dynamic_Gaussian_Processes %%%
% 
% 
% Jilles van Hulst
% 
% 17-02-2023
% 
% 
% Description:
% 1) Extension of Gaussian processes for evolving functions.
% 2) Extension of Kalman filter for infinite-dimensional state (functions).
% This script considers an approximate solution using basis functions which
% allows for recursive and fast computation. The script automatically
% approximates user-defined functions.
% This script considers only the 1D-input case.


%% General Settings

% close all
clear; clc

rng(2)

% plotting settings:
plot_confidence_bounds = 1;     % if set to 1, the covariance estimates are plotted

% define evaluation range:
Ntest = 625;                    % number of points in function estimate evaluation
x_min = -1;                     % minimum input in function estimate evaluation
x_max = 1;                      % maximum input in function estimate evaluation

dx = (x_max-x_min)/(Ntest-1);

xtest = (x_min:dx:x_max).';

% number of observations at each timestep k:
p = 3;

% duration of simulation:
N = 100;

% number of bases:
M = 9;

% basis choice:
% basis = 'RBF';
% basis = 'NRBF';      % normalized version of RBF
basis = 'Fourier';
% basis = 'Discrete';


%% System functions

% evolution kernel:
sigma_kf = 0.07;
a_kf = 0.9/sqrt(2*pi)/sigma_kf;                               % normalized to have area 1 without first product term
kf = @(X,V) a_kf*squexp(X,V,sigma_kf);    
% kf = @(X,V) sin(X(:)*pi)*sin(V(:)*pi).';
% kf = @(X,V) 2*X(:)*(2*(V(:)<=0).'-2*(V(:)>0).');
% kf = @(X,V) eq(X(:),V(:).')/(dx);                             % "integrator dynamics", kroneckerDelta

% initial function mean:
f_bar = @(X) zeros(size(X,1),1);                                  % zero mean
% f_bar = @(X) ones(size(X,1),1);                                   % constant mean
% f_bar = @(X) 10*squexp(X,0,0.05);                                 % bell curve around origin
% f_bar = @(X) -2*X.^2+8;                                           % parabola centered on origin
% f_bar = @(X) eq(X(:),0)/(dx);                                     % impulse centered on origin
% f_bar = @(X) double(abs(X(:))<=0.05)/0.1;                           % approximate impulse centered on origin

% initial function covariance:
a_f = 1;
sigma_f = 0.7;
Q_f = @(X,V) a_f*squexp(X,V,sigma_f);                           % smooth initial condition variance
% Q_f = @(X,V) zeros(size(X,1),size(V,1));                      % no variance in i.c.
% Q_f = @(X,V) sigma_f^2*eq(X(:),V(:).');  	                    % kroneckerDelta: resembles white noise, spatially uncorrelated.
% Q_f = @(X,V) double(abs(X(:)-V(:).')<=0.1);                           % approximate impulse centered on origin

% disturbance covariance:
a_w = 0.35;
sigma_w = 0.15;
% Q_w = @(X,V) a_w*squexp(X,V,sigma_w);                           % smooth disturbances
Q_w = @(X,V) zeros(size(X,1),size(V,1));                      % no disturbances
% Q_w = @(X,V) sigma_w^2*eq(X(:),V(:).');  	                    % kroneckerDelta: resembles white noise, spatially uncorrelated.

% measurement noise covariance:
sigma_v = 0.1;
Q_v = @(X,V) sigma_v^2*eq(X(:),V(:).');                         % kroneckerDelta: resembles white noise, spatially uncorrelated.



%% Ground Truth Basis Functions (always discrete bases)

bounds = x_min:(x_max*(1+1e-12)-x_min)/(Ntest):x_max*(1+1e-12); % +1e-12 added to ensure bounds themselves fall in a base

u_true = cell(Ntest);
Utest_true = NaN(Ntest,Ntest);
Usamp_true = NaN(Ntest,p);

for i=1:Ntest
    u_true{i} = @(X) (double(bounds(i)<=X & X<bounds(i+1)));
    Utest_true(i,:) = u_true{i}(xtest);       % evaluate the basis functions at the test points
end


% fitting of system functions (simple since orthonormal):

Lambda_U_true = Utest_true.'*Utest_true*dx;

Pinv_U_true = (Utest_true.'*Utest_true)\Utest_true.';


% evolution matrix:

Lambda_true = Pinv_U_true*kf(xtest,xtest);


% initial conditions of the function:

z_bar_true = Pinv_U_true*f_bar(xtest);

Lambda_f_true = Pinv_U_true*Q_f(xtest,xtest);


% disturbance covariance matrix:
Lambda_W_true = Pinv_U_true*Q_w(xtest,xtest);


% make all covariance matrices symmetrical:
Lambda_f_true = (Lambda_f_true.'+Lambda_f_true)/2;
Lambda_W_true = (Lambda_W_true.'+Lambda_W_true)/2;



%% Approximation Basis Functions

u = cell(M);
Utest = NaN(M,Ntest);
Usamp = NaN(M,p,N+1);

switch basis
    case 'RBF'
        orthogonal = false;
        if isequal(M,1)
            db = (x_max-x_min)/2;
            center = (x_max+x_min)/2;
        else
            db = (x_max-x_min)/(M-1);
            center = x_min:db:x_max;
        end

        l = db*0.8;                         % tuned lengthscale
        
        for i=1:M
            u{i} = @(X) (squexp(X,center(i),l));
            Utest(i,:) = u{i}(xtest);       % evaluate the basis functions at the test points
        end
    case 'NRBF'
        orthogonal = false;
        if isequal(M,1)
            db = (x_max-x_min)/2;
            center = (x_max+x_min)/2;
        else
            db = (x_max-x_min)/(M-1);
            center = x_min:db:x_max;
        end

        l = db*0.6;                         % tuned lengthscale
        
        Unorm = @(X) sum(squexp(X,center.',l),2);
        
        for i=1:M
            u{i} = @(X) (squexp(X,center(i),l)./Unorm(X));
            Utest(i,:) = u{i}(xtest);       % evaluate the basis functions at the test points
        end

    case 'Fourier'
        orthogonal = true;
        n = 0;
        period = x_max-x_min;
        
        for i=1:M
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
        db = (x_max+1e-12-x_min)/(M);     % +1e-12 added to ensure every point in xtest is in only 1 base.
        bounds = x_min:db:x_max+1e-12;

        for i=1:M
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



%% Fitting of system functions for approximation

timer = tic;


UU_mat = NaN(Ntest,Ntest,M^2);
Lambda_U = NaN(M,M);

for i=1:M
    for j=1:M
        UU_mat(:,:,(i-1)*M+j) = Utest(i,:).'*Utest(j,:);
        Lambda_U(i,j) = sum(Utest(i,:).*Utest(j,:))*dx;
    end
end

UU_mat = reshape(UU_mat,[Ntest^2,M^2]);


if orthogonal
% fitting of system (simplified since orthogonal):
    Pinv_UU = NaN(M^2,Ntest^2);
    for i=1:M^2
        UU_mag = UU_mat(:,i).'*UU_mat(:,i);
        Pinv_UU(i,:) = UU_mat(:,i).'./UU_mag;
    end

else
    Pinv_UU = (UU_mat.'*UU_mat)\(UU_mat.');

end


% evolution matrix:
kkf = reshape(kf(xtest,xtest).',[Ntest^2,1]);           % evaluated kernel matrix

Lambda = reshape(Pinv_UU*kkf,[M,M]);         % least squares fit in basis


% initial conditions of the function:
QQf = reshape(Q_f(xtest,xtest),[Ntest^2,1]);

Lambda_f = reshape(Pinv_UU*QQf,[M,M]);

z_bar = (Utest*Utest.')\Utest*f_bar(xtest);


% disturbance covariance matrices:
QQv = reshape(Q_w(xtest,xtest),[Ntest^2,1]);

Lambda_V = reshape(Pinv_UU*QQv,[M,M]);


fprintf('Fitting approximate DGP took %.1f seconds', toc(timer));



% make all covariance matrices symmetrical:
Lambda_f = (Lambda_f.'+Lambda_f)/2;
Lambda_V = (Lambda_V.'+Lambda_V)/2;

%% Initialize Loop Variables

% 'true' system:

x = NaN(p,N+1);
v = NaN(p,N+1);
y = NaN(p,N+1);
z = NaN(Ntest,N+1);
xi = NaN(Ntest,N+1);
w = NaN(Ntest,N+1);

f = NaN(Ntest,N+1);
f_samp = NaN(p,N+1);


% estimator:

Psi_Gamma = NaN(M,p,N+1);

z_upd = NaN(M,N+1);
z_pred = NaN(M,N+2);

f_upd = NaN(Ntest,N+1);
f_pred = NaN(Ntest,N+2);

Psi_upd = NaN(M,M,N+1);
Psi_pred = NaN(M,M,N+2);

c_upd = NaN(Ntest,Ntest,N+1);
c_pred = NaN(Ntest,Ntest,N+2);

e_upd = NaN(Ntest,N+1);
e_pred = NaN(Ntest,N+1);



%% Initial Conditions

% initial condition of system at k=1:
z(:,1) = mvnrnd(z_bar_true,Lambda_f_true);

% initial condition of estimator:
z_pred(:,1) = z_bar;                    % prior mean estimate
% z_pred(:,1) = zeros(M,1);               % zero prior mean estimate
Psi_pred(:,:,1) = Lambda_f;             % prior covariance estimate

c_pred(:,:,1) = Utest.'*Psi_pred(:,:,1)*Utest;
f_pred(:,1) = Utest.'*z_pred(:,1);


% plot initial function and estimate:

figure(2)
clf
plot(xtest,Utest_true.'*z(:,1),'--k','DisplayName','True function','LineWidth',1)
hold on; grid on
plot(xtest,f_pred(:,1),'r','DisplayName','Mean estimate','LineWidth',1)
if plot_confidence_bounds
CI_max = f_pred(:,1)+2*sqrt(diag(c_pred(:,:,1)));
CI_min = f_pred(:,1)-2*sqrt(diag(c_pred(:,:,1)));
plot(xtest,CI_max,'b:','HandleVisibility','off')
plot(xtest,CI_min,'b:','HandleVisibility','off')
fill([xtest.', fliplr(xtest.')],[CI_max.', fliplr(CI_min.')],'cyan','FaceAlpha',0.3,'DisplayName','Estimated 95% conf.')
end
title('$t=0$')
xlabel('$x$')
ylabel('$f_{t}(x)$')
legend()

pause('Press any key to continue')



%% Simulation Loop

for t=1:N+1

    % measurement points:
    x(:,t) = unifrnd(x_min,x_max,p,1);          % random sample point in \mathcal{X}

    for i=1:Ntest
        Usamp_true(i,:) = u_true{i}(x(:,t));    % evaluate the basis functions at the sample points
    end

    for i=1:M
        Usamp(i,:,t) = u{i}(x(:,t));            % evaluate the basis functions at the sample points
    end

    % evaluate true noiseless function:
    f(:,t) = Utest_true.'*z(:,t);               % at test points
    f_samp(:,t) = Usamp_true.'*z(:,t);          % at sample points

    % measure:
    v(:,t) = mvnrnd(zeros(p,1),Q_v(x(:,t),x(:,t)));
    y(:,t) = f_samp(:,t) + v(:,t);


    % DGP update:
    Psi_Gamma(:,:,t) = Psi_pred(:,:,t)*Usamp(:,:,t)/(Usamp(:,:,t).'*(Psi_pred(:,:,t))*Usamp(:,:,t)+Q_v(x(:,t),x(:,t)));
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
    xi(:,t) = mvnrnd(zeros(size(Lambda_W_true,1),1),Lambda_W_true);
    w(:,t) = Utest_true.'*xi(:,t);
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
    CI_max = f_upd(:,t)+2*sqrt(diag(c_upd(:,:,t)));
    CI_min = f_upd(:,t)-2*sqrt(diag(c_upd(:,:,t)));
    plot(xtest,CI_max,'b:','HandleVisibility','off')
    plot(xtest,CI_min,'b:','HandleVisibility','off')
    fill([xtest.', fliplr(xtest.')],[CI_max.', fliplr(CI_min.')],'cyan','FaceAlpha',0.3,'DisplayName','Estimated 95% CI')
    end
    title(sprintf('$t=%i$',t),'FontSize',15)
    xlabel('$x$','FontSize',15)
    ylabel('$f_{t}(x)$','FontSize',15)
    legend()

    pause(0.15)      % for the sake of visualization

end



%% Plotting

c = winter(N+1);


figure(100)
clf
set(gcf,'Position',[573 544 560 300])
for t=1:N+1
    plot3(xtest,ones(size(xtest))*(t-1),f(:,t),'color',c(t,:),'LineWidth',1.5)

    grid on; hold on

    plot3(x(:,t),ones(size(x,1),1)*(t-1),y(:,t),'x','color',c(t,:),'MarkerSize',15)
end
xlabel('$x$','FontSize',15)
ylabel('$t$','FontSize',15)
zlabel('$f_{t}(x)$','FontSize',15)
xlim([min(xtest) max(xtest)])
ylim([0 N])
view(-45,45)


figure(301)
clf
set(gcf,'Position',[573 544 560 300])
for t=1:N+1
    plot3(xtest,ones(size(xtest))*(t-1),f_upd(:,t),'color',c(t,:),'LineWidth',1.5)

    grid on; hold on

    if plot_confidence_bounds
        CI_max = f_upd(:,t)+2*sqrt(diag(c_upd(:,:,t)));
        CI_min = f_upd(:,t)-2*sqrt(diag(c_upd(:,:,t)));
    
        plot3(xtest,ones(size(xtest))*(t-1),CI_min,'color',c(t,:),'LineWidth',0.5)
        plot3(xtest,ones(size(xtest))*(t-1),CI_max,'color',c(t,:),'LineWidth',0.5)
    end
    plot3(x(:,t),ones(size(x,1),1)*(t-1),y(:,t),'x','color',c(t,:),'MarkerSize',15)
end
xlabel('$x$','FontSize',15)
ylabel('$t$','FontSize',15)
zlabel('$\hat{f}_{t|t}(x)$','FontSize',15)
xlim([min(xtest) max(xtest)])
ylim([0 N])
view(-45,45)


figure(302)
clf
set(gcf,'Position',[573 544 560 300])
for t=1:N+1
    plot3(xtest,ones(size(xtest))*(t-1),f_pred(:,t),'color',c(t,:),'LineWidth',1.5)

    grid on; hold on

    if plot_confidence_bounds
        CI_max = f_pred(:,t)+2*sqrt(diag(c_pred(:,:,t)));
        CI_min = f_pred(:,t)-2*sqrt(diag(c_pred(:,:,t)));
    
        plot3(xtest,ones(size(xtest))*(t-1),CI_min,'color',c(t,:),'LineWidth',0.5)
        plot3(xtest,ones(size(xtest))*(t-1),CI_max,'color',c(t,:),'LineWidth',0.5)
    end
    plot3(x(:,t),ones(size(x,1),1)*(t-1),y(:,t),'x','color',c(t,:),'MarkerSize',15)
end
xlabel('$x$','FontSize',15)
ylabel('$t$','FontSize',15)
zlabel('$\hat{f}_{t|t-1}(x)$','FontSize',15)
xlim([min(xtest) max(xtest)])
ylim([0 N])
view(-45,45)


figure(401)
clf
set(gcf,'Position',[573 544 560 300])
for t=1:N+1
    plot3(xtest,ones(size(xtest))*(t-1),e_upd(:,t),'color',c(t,:),'LineWidth',1.5)

    grid on; hold on

    if plot_confidence_bounds
        CI_max = e_upd(:,t)+2*sqrt(diag(c_upd(:,:,t)));
        CI_min = e_upd(:,t)-2*sqrt(diag(c_upd(:,:,t)));
    
        plot3(xtest,ones(size(xtest))*(t-1),CI_min,'color',c(t,:),'LineWidth',0.5)
        plot3(xtest,ones(size(xtest))*(t-1),CI_max,'color',c(t,:),'LineWidth',0.5)
    end
end
xlabel('$x$','FontSize',15)
ylabel('$t$','FontSize',15)
zlabel('$e_{t|t}(x)$','FontSize',15)
xlim([min(xtest) max(xtest)])
ylim([0 N])
view(-45,30)


figure(500)
set(gcf,'Position',[573 544 560 210])
plot(0:N,sqrt(sum((e_upd).^2,1)),'LineWidth',1,'DisplayName',sprintf('M = %i',M))
grid on; hold on
xlabel('$t$','FontSize',15)
ylabel('$\Vert e_{t|t} \Vert_2$','FontSize',15)
legend('Interpreter','latex')


%% Functions

function K = squexp(U,V,sigma)
    % input U (N by n): 
    % input V (M by n):
    % input sigma (1 by 1): length scale. Equal in every direction
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
