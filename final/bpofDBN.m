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
etaa = 100;
Nbmp = 200;
%% mode
only2 = 0;
loaddata = 0;

if loaddata,
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
elseif ~loaddata,
    wij_wi = rand(nin, nh1) - 0.5;
    wjk_wi = rand(nh1, nh2) - 0.5;
    wkt = rand(nh2, nt) - 0.5;
    wtl = rand(nt, nd) - 0.5;
end

%% monitoring
hj_all = zeros(Nbmp, 1);
hk_all = zeros(Nbmp, 1);
ht_all = zeros(Nbmp, 1);

%% run
for c = 1:Nbmp;
    cc = floor(rand*899);
    di = zeros(10, 1);
    if only2 == 1,
        digit = 2;
        fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',cc);
        % fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',cc);   %for windows
        di(3) = 1;
    elseif only2 == 0,
        digit = floor(rand*10);
        fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',digit,cc);
        % fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',floor(rand*10),cc);
        di(digit + 1) = 1;
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
    hl = (wtl)' * ht_a ./2000 ;      % 1x2000 x 2000x10 '
% error
    er = di - hl;

%% monitoring
    hj_all(c) = sum(sum(hj));
    hk_all(c) = sum(sum(hj));
    ht_all(c) = sum(sum(hj));


    delta_tl = ht_a ./2000 * (er * (-1))';      % 2000x1 x [10x1 .x 10x1] '
    delta_kt = hk_a * (wtl ./2000 * (er .* (-1)) .* (ht_a .* (1 - ht_a)))';      % 500x1 x {2000x10 x 10x1.x [2000x1(1-2000x1)]}'
    delta_jk = hj_a * (wkt * (wtl ./2000 * (er .* (-1)) .* (ht_a .* (1 - ht_a))) .* (hk_a .* (1 - hk_a)))';
    % 500dx1 x{500x2000 x [2000x10 x 10x1 .x (2000x1")].x [500x1(1-500x1)]}'
    delta_ij = vi * (wjk_wi * (wkt * (wtl ./2000 * (er .* (-1)) .* (ht_a .* (1 - ht_a))) .* (hk_a .* (1 - hk_a))) .* (hj_a .* (1 - hj_a)))';
    % 784x1 x {500dx500 x [500x2000 x (2000x10 x (10x1)) .x 2000x1] .x 500x1}'

    wtl = wtl +  etaa .* delta_tl;
    wkt = wkt +  etaa .* delta_kt;
    wjk_wi = wjk_wi + etaa .* delta_jk;
    wij_wi = wij_wi + etaa .* delta_ij;
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
        di_t(3) = 1;
    elseif only2 == 0,
        digit_t = floor(rand*10);
        ftname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',floor(rand*10), tt);
        % fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',floor(rand*10),cc);
        di_t(digit_t + 1) = 1;
    end
    B = double(imread(ftname));
    vt = reshape(B./255, nin, 1);
    hl_t = (sigmoid(sigmoid(sigmoid((vt)' * wij_wi) * wjk_wi) * wkt) * wtl)';
    [value p] = max(hl_t);
    if p == (digit_t + 1)
      accuracy(t) = 1;
    else
      accuracy(t) = 0;
    end
end

fprintf('= %2.1f%%\n', sum(accuracy) / Ntest * 100);

figure(1)
subplot(3, 1, 1); plot(1:Nbmp, hj_all, 'r-'); ylabel('sumhj');
subplot(3, 1, 2); plot(1:Nbmp, hk_all, 'r-'); ylabel('sumhk');
subplot(3, 1, 3); plot(1:Nbmp, ht_all, 'r-'); ylabel('sumht');
