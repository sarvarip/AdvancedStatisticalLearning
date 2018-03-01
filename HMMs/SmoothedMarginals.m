function [ marginal ] = SmoothedMarginals(  A, b, alpha, beta, T, Nhidden  )
%SMOOTHEDMARGINALS Computes the two-slice smoothed marginals from alpha,
% beta, A and b.

marginal = zeros(Nhidden, Nhidden, T-1);

for t=1:T-1
    marginal(:, :, t) = normalize(A .* (alpha(:, t) * (b(:, t+1) .* beta(:, t+1))'));
end

