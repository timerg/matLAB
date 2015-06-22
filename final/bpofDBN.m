clear all;
close all;

nin = 784;
nh1 = 500;
nh2 = 500;
nt = 2000;
nd = 10;
%% bias
b0 = 0;
b1 = 0;
b2 = 0;
b3 = 0;
b4 = 0;
%% parameters
etaa = 0.001;
Nbmp = 1;
%% mode
only2 = 0;

load('~/GitHub/matLAB/final/weight_ij', 'wij_wi')
load('~/GitHub/matLAB/final/weight_jk', 'wjk_wi')
load('~/GitHub/matLAB/final/weight_kt', 'wkt')
load('~/GitHub/matLAB/final/weight_tl', 'wtl')
load('~/GitHub/matLAB/final/hidden_j', 'hj')
load('~/GitHub/matLAB/final/hidden_k', 'hk')
load('~/GitHub/matLAB/final/hidden_t', 'ht')
load('~/GitHub/matLAB/final/hidden_l', 'hl')

% load('/Users/timer/Documents/gitHub/matLAB/final/weight_ij', 'wij_wi')
% load('~/GitHub/matLAB/final/weight_jk', 'wjk_wi')
% load('~/GitHub/matLAB/final/weight_kt', 'wkt')
% load('~/GitHub/matLAB/final/weight_tl', 'wtl')
% load('~/GitHub/matLAB/final/hidden_j', 'hj')
% load('~/GitHub/matLAB/final/hidden_k', 'hk')
% load('~/GitHub/matLAB/final/hidden_t', 'ht')
% load('~/GitHub/matLAB/final/hidden_l', 'hl')


for c = 1:Nbmp;
    cc = ceil(rand*999);
    di = zeros(10, 1);
    if only2 == 1,
        digit = 2;
        fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',cc-1);
        % fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',cc);   %for windows
        di(3) = 10;
    elseif only2 == 0,
        digit = floor(rand*10);
        fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',digit,cc-1);
        % fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',floor(rand*10),cc);
        di(digit + 1) = 10;
    end
    A = double(imread(fname));
    vi = reshape(A./255, nin, 1);
% hj
    hj = (wij_wi)' * vi;     % 1x784 x 784x500 '
    hj_a = sigmoid(hj);
% hk
    hk = (wjk_wi)' * hj_a;     % 1x500 x 500x500 '
    hk_a = sigmoid(hk);
% ht
    ht = (wkt)' * hk_a;      % 1x500 x 500x2000 '
    ht_a = sigmoid(ht);
% hl
    hl = (wtl)' * ht_a;      % 1x2000 x 2000x10 '
    hl_a = sigmoid(hl);
% error
    er = di - hl_a;

    delta_tl = ((er * (-1) .* hl_a .* (1 - hl_a)) * (ht_a)')';      % 10x1 x 1x2000 '
    delta_kt = hk_a * (wtl * (er .* (-1) .* hl_a .* (1 - hl_a)) .* (ht_a .* (1 - ht_a)))';      % 500x1 x {2000x10 x 10x1.x [2000x1(1-2000x1)]}'
    delta_jk = hj_a * (wkt * (wtl * ((er .* (-1) .* hl_a .* (1 - hl_a)) .* hl_a .* (1 - hl_a)) .* (ht_a .* (1 - ht_a))) .* (hk_a .* (1 - hk_a)))';
    % 500dx1 x{500x2000 x [2000x10 x 10x1 .x (2000x1")].x [500x1(1-500x1)]}'
    delta_ij = vi * (wjk_wi * (wkt * (wtl * (er .* (-1) .* hl_a .* (1 - hl_a)) .* (ht .* (1 - ht))) .* (hk_a .* (1 - hk_a))) .* (hj_a .* (1 - hj_a)))';
    % 784x1 x {500dx500 x [500x2000 x (2000x10 x (10x1)) .x 2000x1] .x 500x1}'

    wij_wi = wij_wi + etaa .* delta_ij;
    wjk_wi = wjk_wi + etaa .* delta_jk;
    wkt = wkt +  etaa .* delta_kt;
    wtl = wtl +  etaa .* delta_tl;
end

%% testing
Ntest = 100;
accuracy = zeros(Ntest,1);
for t = 1:Ntest
    tt = 999 - Ntest + t;
    dt_t = zeros(10, 1);
    if only2 == 1,
        digit_t = 2;
        ftname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp', tt);
        % fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',cc);   %for windows
        di_t(3) = 10;
    elseif only2 == 0,
        digit_t = floor(rand*10);
        ftname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',floor(rand*10), tt);
        % fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',floor(rand*10),cc);
        di_t(digit_t + 1) = 10;
    end
    B = double(imread(ftname));
    vt = reshape(B./255, nin, 1);
    hl_t = (sigmoid(sigmoid(sigmoid(sigmoid(sigmoid((vt)') * wij_wi) * wjk_wi) * wkt) * wtl))';
    [value p] = max(hl_t);
    if p == (digit_t + 1)
      accuracy(t) = 1;
    else
      accuracy(t) = 0;
    end
end

fprintf('= %2.1f%%\n', sum(accuracy));
