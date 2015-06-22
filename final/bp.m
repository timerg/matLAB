close all
clear all

nin = 784;
nh1 = 20;
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
etaij = 0.5;
etajl = 0.01;
Nbmp = 1000;
%% mode
only2 = 0;
loaddata = 0;

wj_ini = importdata('~/GitHub/matlab/hw4/wj_ini.mat')
wk_ini = importdata('~/GitHub/matlab/hw4/wk_ini.mat')
wij_wi = ceil(wj_ini .* 10^ 3) ./ 10^ 3;
wjl = ceil(wk_ini .* 10^ 3) ./ 10^ 3;



% wij_wi = rand(nin, nh1) .* 0.2;
% wjk_wi = rand(nh1, nh2) - 0.5;
% wkt = rand(nh2, nt) - 0.5;
% wtl = rand(nt, nd) - 0.5;

%% temp
% wjl = rand(nh1, nd) .* 0.2;

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
  hl = ((hj_a)' * wjl)' ./ nh1;
  er = zeros(10, 1);
  er = di - hl;

  delta_ij = vi * sum((wjl .* (dsigmoid(hj_a) * (er)'))');
  delta_jl = hj_a * (er)';

  wij_wi = wij_wi + delta_ij .* etaij;
  wjl = wjl + delta_jl .* etajl;

end




%% test
Ntest = 100;
confusion = zeros(100,100);
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
    hl_t = sigmoid((vt)' * wij_wi) * wjl;
    [y, p] = max(hl_t);
    confusion(digit_t + 1, p) = confusion(digit_t + 1, p) + 1;
  end
end

accuracy = 100*sum(diag(confusion))/sum(sum(confusion));
fprintf('Total Accuracy = %2.1f%%\n',accuracy);
