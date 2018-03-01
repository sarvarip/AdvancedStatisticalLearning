load('data_multinomial.mat');
load('init_multinomial.mat');

[ A, B, pi ] = EM_estimate_discrete(Y_d, 2, 100, 1e-6, Init_Multinomial);
[ S ] = ViterbiDecode(Y_d, 2, 'multinomial', Init_Multinomial);

fprintf('\n*** Viterbi decoding accuracy: %f ***\n', sum(sum(S == S_d)) / numel(S_d));