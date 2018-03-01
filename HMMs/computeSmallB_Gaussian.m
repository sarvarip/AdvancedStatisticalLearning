function [ b ] = computeSmallB_Gaussian(Y, Means, Variances, Nhidden, T)
%COMPUTESMALLB_GAUSSIAN Compute the probabilities for the data points Y for
% a Gaussian observation model with parameters Means and Variances.
%         
%         Input parameters:
%             - Y: the data
%             - Means: vector of the current estimates of the means
%             - Variances: vector of the current estimates of the variances
%             - Nhidden: number of hidden states
%             - T: length of the sequence
%         Output:
%             - b: vector of observation probabilities
% 
% Vectorized computation of the probabilities can be several times times as 
% fast as a for loop!

%b cannot be a VECTOR, it has to be a MATRIX, which stores the vectors for
%all observations in one sequence, since we pass it to the backward-forward
%algorithm
%Since b is a matrix, Y cannot be a data POINT, it has to be the VECTOR of
%observations in one sequence 

b = zeros(Nhidden, T);
for i = 1:Nhidden
    b(i,:) = 1/sqrt(2*pi*Variances(i))*exp(-(Y-Means(i)).^2/(2*Variances(i))); 
end


end