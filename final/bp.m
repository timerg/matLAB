% close all
clear all

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
etaij = 0.01;
etatl = 0.1;
etajt = 0.01;
Nbmp = 800;
%% mode
only2 = 0;
loaddata = 0;


wij_wi = rand(nin, nh1) .* 0.2;
% wjk_wi = rand(nh1, nh2) - 0.5;
% wkt = rand(nh2, nt) - 0.5;
wtl = rand(nt, nd) - 0.5;

%% temp
wjt = rand(nh1, nt) .* 0.2;

%%supervise
er_all = zeros(nd, Nbmp);
%% train
for c = 1:Nbmp;
  c
  cc = floor(rand * 699);
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
  hj = ((vi)' * wij_wi)' ./ nin;
  hj_a = sigmoid(hj);
  ht = ((hj_a)' * wjt)' ./nh1 .* 10;
  ht_a = sigmoid(ht);
  hl = ((ht_a)' * wtl)' ./ nt .* 100;
  er = zeros(10, 1);
  er = di - hl;
  er_all(:, c) = er;

  delta_tl = ht_a * (er)';
  delta_jt_temp = sum((wtl .* (dsigmoid(ht_a) * (er)'))');
  delta_jt = hj_a * delta_jt_temp;
  delta_ij = vi * sum((wjt .* (dsigmoid(hj_a) * delta_jt_temp))');

  % delta_tl = ht_a * (er)';
  % delta_kt_temp = sum((wtl .* (dsigmoid(ht_a) * (er)'))')
  % delta_kt = hk_a * delta_kt_temp;
  % delta_jk_temp = sum((wkt .* (dsigmoid(hk_a) * delta_kt_temp))')
  % delta_jk = hj_a * delta_jk_temp;
  % delta_ij = vi * sum(wjk .* (dsigmoid(hj_a) * delta_jk_temp)')


  wij_wi = wij_wi + delta_ij .* etaij;
  wjt = wjt + delta_jt .* etajt;
  wtl = wtl + delta_tl .* etatl;
end

figure(1)
plot(1:Nbmp, sum(er_all.^2), 'r-');




%% test
Ntest = 100;
confusion = zeros(10,10);
for t = 1:Ntest
  tt = 999 - Ntest + t;
  for digit_t = 0:9
    if only2 == 1,
        digit_t = 2;
        ftname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp', tt);
        % fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',cc);   %for windows
    elseif only2 == 0,
        ftname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp', digit_t, tt);
        % fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',floor(rand*10),cc);
    end
    B = double(imread(ftname));
    vt = reshape(B ./ 255, nin, 1);
    hl_t = sigmoid(sigmoid((vt)' * wij_wi) * wjt) * wtl;
    [y, p] = max(hl_t);
    confusion(digit_t + 1, p) = confusion(digit_t + 1, p) + 1;
  end
end

accuracy = 100*sum(diag(confusion))/sum(sum(confusion));
fprintf('Total Accuracy = %2.1f%%\n',accuracy);
