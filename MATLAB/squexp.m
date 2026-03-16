function K = squexp(U,V,sigma)
    % SQUEXP  Squared exponential (RBF/Gaussian) kernel matrix.
    %
    %   K = squexp(U, V, sigma)
    %
    %   Inputs:
    %     U     (N by d) matrix of N points in d dimensions
    %     V     (M by d) matrix of M points in d dimensions
    %     sigma (scalar)  length scale, assumed equal in every direction
    %
    %   Output:
    %     K     (N by M) matrix with K(i,j) = exp(-||U(i,:)-V(j,:)||^2 / (2*sigma^2))

    D = sum(U.^2,2) + sum(V.^2,2).' - 2*(U*V.');
    K = exp(-D/(2*sigma^2));
end
