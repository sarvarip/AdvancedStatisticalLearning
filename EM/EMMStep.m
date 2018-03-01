% to be filled in

function A = EMMStep(B,C,D)

mus = zeros(size(B,2), C); 
coeffs = zeros(C, 1);
covars = zeros(size(B,2), size(B,2), C);
for cluster_number = 1:C
    Nk = sum(D(:,cluster_number)); %Bishop book Eq (9.27)
    %Eq (9.24)
    mus(:,cluster_number) = (1/Nk * sum(bsxfun(@times, B, D(:,cluster_number)),1))';
    %Eq (9.26)
    coeffs(cluster_number) = Nk/size(B,1);
    %Eq (9.25)
    temp = zeros(size(B,2), size(B,2));
    for sample = 1:size(B,1)
        temp = temp + D(sample, cluster_number)*((B(sample,:))'-mus(:,cluster_number))*(B(sample,:)-(mus(:,cluster_number))');
    end
    covars(:,:,cluster_number) = 1/Nk * temp;
end

%Wrapping the parameters
A.means = {mus(:,1), mus(:,2), mus(:,3), mus(:,4)};
A.covar = {covars(:,:,1), covars(:,:,2), covars(:,:,3), covars(:,:,4)};
A.mixCoeff = {coeffs(1), coeffs(2), coeffs(3), coeffs(4)};

end

