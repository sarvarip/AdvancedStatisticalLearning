function [ A, B, pi ] = EM_estimate_discrete(Y, Nhidden, Niter, ...
    epsilon, init)
%EM_ESTIMATE_DISCRETE EM algorithm for an HMM with discrete observations.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Useful quantities
N = size(Y,1);
T = size(Y,2);

% In the maximization step for B you will have to compute a quantity 
% involving indicators on the values of Y. One efficient way to do it is to
% pre-compute a representation of Y using one-hot encoding. In MATLAB:

% % X sparse coding
% Nv = length(unique(Y));
% X = zeros(T, Nv);
% for i=1:T
%     X(i, Y(i)) = 1;
% end
% % Maximization: emission matrix
% B1 = B1 + gamma * X;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization of the parameters

% Initial transition matrix must be stochastic (rows sum to 1)
A = init.A;

% Observation matrix B
B = init.B;

% Class prior
pi = init.pi;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EM algorithm
i=0;
% Initialize convergence criterion here

Nv = length(unique(Y));
prev_logprob = -inf;
logprob = -10^10;

while i<Niter && logprob - prev_logprob > epsilon
    
    prev_logprob = logprob;
    
    %Expectation step
    gamma = 0;
    marginal = 0;
    logprob = 0;
    evidence_max_nominator = zeros(Nhidden, Nv);
    evidence_max_nominator_temp = zeros(Nhidden, Nv);
    for l = 1:N %no. of sequences
        b = computeSmallB_Discrete(Y, B);
        [ alpha, beta, gamma_temp, logprob_temp, ~ ] = ForwardBackwardSmoothing( A, b, ...
        pi, Nhidden, T );
        marginal_temp = SmoothedMarginals(  A, b, alpha, beta, T, Nhidden );
        for j = 1:Nv
            evidence_max_nominator_temp(:,j) = sum(bsxfun(@times, gamma_temp, (Y(l,:)==j)),2);
        end
        
        marginal = marginal + marginal_temp;
        gamma = gamma + gamma_temp;
        logprob = logprob + logprob_temp;
        evidence_max_nominator = evidence_max_nominator + evidence_max_nominator_temp;
    end
           
    %Maximization step
    pi = gamma(:,1)/sum(gamma(:,1)); %not gonna be N, because beta was not normalized properly and we are working with ratios. In addition, we already summed across the sequences!
    sum_marg = sum(marginal,3);
    A = bsxfun(@rdivide, sum_marg, sum(sum_marg,2));
    B = bsxfun(@rdivide, evidence_max_nominator, sum(gamma, 2));
    
    disp(logprob);
    
end

end
