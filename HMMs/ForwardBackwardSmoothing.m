function [ alpha, beta, gamma, lp, Z ] = ForwardBackwardSmoothing( A, b, ...
    pi, N, T )
%FORWARDBACKWARDSMOOTHING Smoothing using the forward-backward algorithm
%   Section 17.4.x of K. Murphy's book
% Input:
%   - A: estimated transition matrix
%   - b: local evidence vector (observation probabilities)
%   - pi: initial distribution of states
%   - N: number of hidden states
%   - T: length of the sequence
% Output:
%   - alpha: filtered belief state as defined in ForwardFiltering
%   - beta: conditional likelihood of future evidence as defined in
%   BackwardFiltering
%   - gamma: gamma_t(j) proportional to alpha_t(j) * beta_t(j)
%   - lp: log probability defined in ForwardFiltering
%   - Z: constant defined in ForwardFiltering

[alpha, lp, Z] = ForwardFiltering( A, b, pi, N, T );
beta = BackwardFiltering(A, b, N, T);
gamma = alpha.*beta;

end