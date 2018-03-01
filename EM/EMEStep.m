% to be filled in

function D = EMEStep(B,C,A)

mus = zeros(size(B,2),C);
covars = zeros(size(B,2),size(B,2),C);
coeffs = zeros(1,C);

for i = 1:4 %Unwrapping the parameters
    mus(:,i) = A.means{i};
    covars(:,:,i) = A.covar{i};
    coeffs(i) = A.mixCoeff{i};
end

temp = zeros(size(B,1), C);
for cluster_number = 1:C %calculating pi_k * N(x|mu_k,sigma_k); vectorized implementation (for speed)
    temp(:, cluster_number) = coeffs(cluster_number)*multivariateGaussian(B, mus(:, cluster_number), covars(:,:,cluster_number));
end

denominator = sum(temp, 2);
gamma = bsxfun(@rdivide, temp, denominator); %calculating the fraction on slide 23, lecture 2, vectorized implementation

D = gamma;

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