function [ b ] = computeSmallB_Discrete(Y, B)
%COMPUTESMALLB_DISCRETE Compute the probabilities for the data points Y 
% for a multinomial observation model with observation matrix B
% 
%         Input parameters:
%             - Y: the data
%             - B: matrix of observation probabilities
%         Output:
%             - b: vector of observation probabilities

%b cannot be a VECTOR, it has to be a MATRIX, which stores the vectors for
%all observations in one sequence, since we pass it to the backward-forward
%algorithm
%Since b is a matrix, Y cannot be a data POINT, it has to be the VECTOR of
%observations in one sequence 

b = B(:,Y);

end