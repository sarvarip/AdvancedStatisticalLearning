load('data_gaussian.mat');
load('init_gaussian.mat');

[ A, Means, Variances, pi ] = EM_estimate_gaussian(Y_c, 2, 100, 1e-6, Init_Gaussian);
[ S ] = ViterbiDecode(Y_c, 2, 'gauss', Init_Gaussian);

fprintf('\n*** Viterbi decoding accuracy: %f ***\n', sum(sum(S == S_c)) / numel(S_c));