function p = logLikelihoodGaussianMixture(A, C, x)
%mus are the means of the K multivariate Gaussians (D*K), where D is the
%Diension and K is the number of Gaussians
%coeffs are the mixing coefficients, size is K*1
%covars are the covariance matrices of the K multivariate Gaussians (D*D*K)
%x are the samples matrix (M*D), where M is the number is samples
%see Bishop on the calculation of the log likelihood

p = 0;
tempo = 0;

mus = zeros(size(x,2), C);
covars = zeros(size(x,2),size(x,2),C);
coeffs = zeros(1,C);

for i = 1:4
    mus(:,i) = A.means{i};
    covars(:,:,i) = A.covar{i};
    coeffs(i) = A.mixCoeff{i};
end

for sample = 1:size(x, 1)
    for cluster_number = 1:size(mus, 2)
        %size(x(sample,:))
        %size(mus(:,cluster_number))
        %size(mus, 2)
        %size(coeffs(cluster_number))
        tempo = tempo + coeffs(cluster_number)*multivariateGaussian(x(sample,:), mus(:,cluster_number), covars(:,:,cluster_number));
    end
    p = p + log(tempo); %has to be log otherwise cannot just add up
    tempo = 0;
end

end

function p = multivariateGaussian(X, mu, Sigma2)
%    p = MULTIVARIATEGAUSSIAN(X, mu, Sigma2) Computes the probability 
%    density function of the examples X under the multivariate gaussian 
%    distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is
%    treated as the covariance matrix. If Sigma2 is a vector, it is treated
%    as the \sigma^2 values of the variances in each dimension (a diagonal
%    covariance matrix)
%    X should have dimension n by k, where n is the number of samples and k
%    is the number of features (base functions)
%    Normalized distribution and vectorized implementation

k = length(mu); %Dimension of the Gaussian

if (size(Sigma2, 2) == 1) || (size(Sigma2, 1) == 1)
    Sigma2 = diag(Sigma2);
end

X = bsxfun(@minus, X, mu(:)'); %(:) makes mu a column vector so it does not matter if input is row or column vector
p = (2 * pi) ^ (- k / 2) * det(Sigma2) ^ (-0.5) * ...
    exp(-0.5 * sum(bsxfun(@times, X * pinv(Sigma2), X), 2));

end