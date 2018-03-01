% to be filled in

function A = GMMInit(B,C)

%random parameters - comment out, see answers.pdf

% [mu, ~] = kMeansInitCentroids(B, C);
% covar_mat = zeros(size(B,2),size(B,2),C);
% for i = 1:C
%     covar_mat(1,1,i) = rand+0.5; %eye(size(B,2))
%     covar_mat(2,2,i) = rand+0.5;
%     saved_rand = rand;
%     covar_mat(1,2,i) = saved_rand-0.5;
%     covar_mat(2,1,i) = saved_rand-0.5;
% end
% mixing_coeff = 1/C*ones(1,C);

%kmeans initialization

no_trials = 10;
totalsumofsquares = zeros(1,no_trials);
classes = zeros(size(B,1),no_trials);
mu = zeros(C,size(B,2),no_trials);

for trial = 1:no_trials %different trials for kmeans, best solution is which has the smallest total sum of squares
    [centroids, ~] = kMeansInitCentroids(B, C);
    [classes(:,trial), mu(:,:,trial), totalsumofsquares(trial)] = fastkmeans(B, centroids, 20);
end

[~,idx] = min(totalsumofsquares); %best k-means solution
%only keep parameters corresponding to the best solution
classes = classes(:,idx);
mu = mu(:,:,idx);
covar_mat = zeros(size(B,2),size(B,2),C);
mixing_coeff = zeros(1,C);

%estimation of covariance matrix and mixing coefficients for each class
for cluster = 1:C
    x_c = B(classes==cluster,:);
    Z = bsxfun(@minus, x_c, mu(cluster,:)); %subtracting the mean
    covar_mat(:,:,cluster) = 1/(size(Z,1)-1) .* (Z'*Z); 
    mixing_coeff(cluster) = size(x_c,1)/size(B,1);
end

%Wrap parameters

A.means = {mu(1,:)', mu(2,:)', mu(3,:)', mu(4,:)'};
A.covar = {covar_mat(:,:,1), covar_mat(:,:,2), covar_mat(:,:,3), covar_mat(:,:,4)};
A.mixCoeff = {mixing_coeff(1), mixing_coeff(2), mixing_coeff(3), mixing_coeff(4)};

end

function [centroids, randidx] = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%idea from Prof. Andrew Ng's Coursera course
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X

% Initialize the centroids to be random examples
% Randomly reorder the indices of examples 
randidx = randperm(size(X, 1)); 
% Take the first K examples as centroids 
centroids = X(randidx(1:K), :);

end

function [class_vec, mean_vec, totalsumofsquares] = fastkmeans(vec, initial_means, maxiterkmeans)
%Kmeans algorithm implementation
%Cluster 1 is first row vector in initial_means
mean_vec = initial_means;
mean_minus = zeros(size(initial_means));
index = 0;
K = size(initial_means, 1);
while ~isequal(mean_minus, mean_vec);
    mean_minus = mean_vec;
    res_matrix = bsxfun(@plus, sum(vec.^2, 2), bsxfun(@minus, (sum(mean_vec.^2, 2))', 2*vec*mean_vec')); 
    %Vectorization: I want (SAMPLEi - MUx)^2, 
    %I calculate SAMPLEi^2+MUx^2-2*SAMPLEi*MUx
    [sumofsquares, ix] = min(res_matrix, [], 2);
    class_vec = ix; %assign the data points to the cluster centre closest 
    %to them
    totalsumofsquares = sum(sumofsquares);
    for class = 1:K
        mean_vec(class,:) = mean(vec(class_vec==class, :), 1);
    end
    index = index + 1;
    %disp(index);
    if index > maxiterkmeans
        break
    end
end

end