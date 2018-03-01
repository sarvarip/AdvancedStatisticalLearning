function [ A, Means, Variances, pi ] = EM_estimate_gaussian(Y, Nhidden, ...
    Niter, epsilon, init)
%EM_ESTIMATE_GAUSSIAN EM algorithm for an HMM with Gaussian observations.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Useful quantities
N = size(Y,1);
T = size(Y,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization of the parameters

% Initial transition matrix should be stochastic (rows sum to 1)
A = init.A;

% Initial means and variances of the emission probabilities

Means = init.Means;
Variances = init.Variances;

% Class prior
pi = init.pi;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EM algorithm

i=0;
% Initialize convergence criterion here

prev_logprob = -inf;
logprob = -10^10;

while i<Niter && logprob - prev_logprob > epsilon
    
    prev_logprob = logprob;
    
    %Expectation step
    gamma = 0;
    marginal = 0;
    logprob = 0;
    mu_max_nominator = zeros(Nhidden,1);
    sigma_max_nominator = zeros(Nhidden,1);
    for l = 1:N %no. of sequences
        b = computeSmallB_Gaussian(Y(l,:), Means, Variances, Nhidden, T);
        [ alpha, beta, gamma_temp, logprob_temp, ~ ] = ForwardBackwardSmoothing( A, b, ...
        pi, Nhidden, T );
        marginal_temp = SmoothedMarginals(  A, b, alpha, beta, T, Nhidden );
        mu_max_nominator_temp = sum(bsxfun(@times, gamma_temp, Y(l,:)),2);
        sigma_max_nominator_temp = sum(gamma_temp.*(bsxfun(@minus, Y(l,:), Means)).^2,2);
        
        marginal = marginal + marginal_temp;
        gamma = gamma + gamma_temp;
        logprob = logprob + logprob_temp;
        mu_max_nominator = mu_max_nominator + mu_max_nominator_temp;
        sigma_max_nominator = sigma_max_nominator + sigma_max_nominator_temp;
    end
           
    %Maximization step
    pi = gamma(:,1)/sum(gamma(:,1)); %not gonna be N, because beta was not normalized properly and we are working with ratios. In addition, we already summed across the sequences!
    sum_marg = sum(marginal,3);
    A = bsxfun(@rdivide, sum_marg, sum(sum_marg,2));
    Means = mu_max_nominator./sum(gamma, 2);
    Variances = sigma_max_nominator./sum(gamma, 2);
    
    disp(logprob);
    
end

end