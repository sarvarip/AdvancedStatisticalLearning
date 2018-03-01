function [ S ] = ViterbiDecode( Y, Nhidden, type, init )
%VITERBIDECODE Perform Viterbi decoding on the smoothed data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Useful quantities
N = size(Y,1);
T = size(Y,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EM and setting the way to compute the vector b according to the type

if strcmp(type, 'gauss')
    [A, Mu, Sigma, Pi] = EM_estimate_gaussian(Y, Nhidden, 100, 1e-6, init);
    SmallB = @(X) computeSmallB_Gaussian(X, Mu, Sigma, Nhidden, length(X));
elseif strcmp(type, 'multinomial')
    [A, B, Pi] = EM_estimate_discrete(Y, Nhidden, 100, 1e-6, init);
    SmallB = @(X) computeSmallB_Discrete(X, B);
else
    error 'Invalid type: must be either gauss or multinomial'
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Viterbi algorithm

Nhidden = length(Pi);
S = zeros(N,T);

for n = 1:N

    %Initialization
    
    delta = zeros(Nhidden, T);
    alpha_state = zeros(Nhidden, T);
    delta(:,1) = log(pi)+log(SmallB(Y(n,1)));
    alpha_state(:,1) = delta(:,1);
    
    %Elongation
    
    for t = 2:T
        [delta(:,t), alpha_state(:,t)] = max(bsxfun(@plus, bsxfun(@plus, log(A'), delta(:,t-1)'), log(SmallB(Y(n,t)))),[],2);
    end

    %Termination
    
    [~, S(n,T)] = max(delta(:,T));
    
    %Traceback
    
    for t = T-1:-1:1
        S(n,t) = alpha_state(S(n,t+1),t+1); %(17.78) Murphy
    end
    
end

end