function [ beta ] = BackwardFiltering(A, b, N, T)
% Perform backward filtering
% Input:
%   - A: estimated transition matrix
%   - b: local evidence vector (observation probabilities)
%   - N: number of hidden states
%   - T: length of the sequence
% Output:
%   - beta s.t beta_t(j) = p(x_t+1:T | z_t = j)

beta = zeros(N,T);
beta(:,end) = ones(N,1);

for i=T-1:-1:1
    beta(:,i) = normalize(A*(beta(:,i+1).*b(:,i+1)));
end

end