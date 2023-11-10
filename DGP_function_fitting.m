%%% DGP_data %%%
% 
% 
% Jilles van Hulst
% 
% 10-11-2023
% 
% 
% Description:
% Script that fits the approximate DGP framework to the system equations
% defined in 'Dynamic_Gaussian_Process_Main.m'. We fit the bases to the
% functions evaluated at a set of inputs 'x_fit' using linear least squares.
% The resulting parameters are gathered in the matrices
%   Lambda, Lambda_U, Lambda_f, Lambda_v
% and the vector
%   z_bar



%% Approximation Basis Functions


% number of equally spaced points used to fit the system functions:
N_fit = 501;


% create equally spaced set of points in function domain:
dx = (x_max-x_min)/(N_fit-1);
x_fit = (x_min:dx:x_max).';


% initialize basis function variables:
u = cell(M,1);              % array of basis functions
U_fit = NaN(M,N_fit);       % basis functions evaluated at fitting points (for approximating)
U_test = NaN(M,N_test);     % basis functions evaluated at testing points (for plotting)
U_samp = NaN(M,p,N_test);   % basis functions evaluated at measurement points (for estimation)


% create basis functions:
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

        l = db*0.8;
        
        for i=1:M
            u{i} = @(X) (squexp(X,center(i),l));
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

        l = db*0.6;
        
        Unorm = @(X) sum(squexp(X,center.',l),2);
        
        for i=1:M
            u{i} = @(X) (squexp(X,center(i),l)./Unorm(X));
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
        end

    case 'Discrete'
        orthogonal = true;
        db = (x_max+1e-12-x_min)/(M);     % +1e-12 added to ensure every point in x_test is in only 1 base.
        bounds = x_min:db:x_max+1e-12;

        for i=1:M
            u{i} = @(X) (double(bounds(i)<=X & X<bounds(i+1))/sqrt(db));
        end
end


% evaluate the basis functions:
for i=1:M
    U_fit(i,:) = u{i}(x_fit);
    U_test(i,:) = u{i}(x_test);
end


% plot the basis functions:
if plot_basis_functions
    figure(10)
    clf
    hold on; grid on
    plot(x_test,U_test)
    xlabel('$x$')
    ylabel('$u_i(x)$')
end


%% System projection


tic

% initialize variables for evaluated basis functions:
U_mat = NaN(N_fit,N_fit,M^2);
Lambda_U = NaN(M,M);

for i=1:M
    for j=1:M
        U_mat(:,:,(i-1)*M+j) = U_fit(i,:).'*U_fit(j,:);
        Lambda_U(i,j) = sum(U_fit(i,:).*U_fit(j,:))*dx;
    end
end

U_mat = reshape(U_mat,[N_fit^2,M^2]);

if orthogonal
% fitting of system (simplified since orthonormal):
    Pinv_U = diag(1./sum(U_mat.^2,1))*(U_mat.');

else
    Pinv_U = (U_mat.'*U_mat)\(U_mat.');

end


% evolution matrix:
Kf = reshape(kf(x_fit,x_fit).',[N_fit^2,1]);

Lambda = reshape(Pinv_U*Kf,[M,M]);


% initial conditions of the function:
QQf = reshape(Q_f(x_fit,x_fit),[N_fit^2,1]);

Lambda_f = reshape(Pinv_U*QQf,[M,M]);

z_bar = (U_fit*U_fit.')\U_fit*m(x_fit);


% disturbance covariance matrices:
QQv = reshape(Q_w(x_fit,x_fit),[N_fit^2,1]);

Lambda_w = reshape(Pinv_U*QQv,[M,M]);


% make all covariance matrices symmetrical in case they are not:
Lambda_f = (Lambda_f.'+Lambda_f)/2;
Lambda_w = (Lambda_w.'+Lambda_w)/2;


toc


if max(abs(eig(Lambda*Lambda_U)))<1
    fprintf('Approximation is stable \n')
else
    fprintf('Approximation is NOT stable \n')
end



%% Functions


function U = Fourier_sin(X,n,period)
    % input X: (N by 1)
    %
    % output U: (N by 1)
    
    U = sin(2*pi*n*X(:)/period);

end

function U = Fourier_cos(X,n,period)
    % input X: (N by 1)
    %
    % output U: (N by 1)
    
    U = cos(2*pi*n*X(:)/period);

end

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