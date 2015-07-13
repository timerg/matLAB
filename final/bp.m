close all
clear all

nin = 784;
nh1 = 100;
nh2 = 500;
nt = 20;
nd = 10;
%% bias
b0 = 0;
b1 = 0;
b2 = 0;
b3 = 0;
b4 = 0;
%% parameters
etaij = 0.05;
etatl = 0.05;
etajt = 0.05;
Nbmp = 1000;
%% mode
only2 = 0;
loaddata = 0;


wij_wi = ceil(rand(nin, nh1) .* 2000) ./ 1000; wij_ini = wij_wi;
% wjk_wi = rand(nh1, nh2) - 0.5;
% wkt = rand(nh2, nt) - 0.5;
wtl = ceil(rand(nt, nd) .* 2000) ./ 1000 ; wtl_ini = wtl;

%% temp
wjt = ceil(rand(nh1, nt) .*2000)./1000; wjt_ini = wjt;

%%supervise
er_all = zeros(nd, Nbmp);
%% train
for c = 1:Nbmp;
  c
  cc = floor(rand * 699);
  di = zeros(1, 10);
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
  vi = reshape(A./255, 1, nin);
  hj = vi * wij_wi ./ nin .* 20;
  hj_a = sigmoid(hj);
  ht = hj_a * wjt ./nh1 .* 10;
  ht_a = sigmoid(ht);
  hl = ht_a * wtl ./ nt .* 100;
  hl_a = sigmoid(hl);
  er = zeros(1, 10);
  er = di - hl;

  delta_tl = (ht_a)' * er;
  delta_jt_temp = wtl * er' .* dsigmoid(ht_a');  % tx1
  delta_jt = (hj_a)' * (delta_jt_temp)';
  % delta_ij = (vi)' * ((wjt * delta_jt_temp)' .* dsigmoid(hj_a));

  delta_ij = (vi)' * ((wjt * (wtl * er' .* dsigmoid(ht_a')))' .* dsigmoid(hj_a));

% delta_ij = (vi)' * sum(er.* ((sum(wjt) .* dsigmoid(ht_a)) * wtl));


  % delta_tl = ht_a * (er)';
  % delta_kt_temp = sum((wtl .* (dsigmoid(ht_a) * (er)'))')
  % delta_kt = hk_a * delta_kt_temp;
  % delta_jk_temp = sum((wkt .* (dsigmoid(hk_a) * delta_kt_temp))')
  % delta_jk = hj_a * delta_jk_temp;
  % delta_ij = vi * sum(wjk .* (dsigmoid(hj_a) * delta_jk_temp)')


  wij_wi = wij_wi + delta_ij .* etaij;
  wjt = wjt + delta_jt .* etajt;
  wtl = wtl + delta_tl .* etatl;

%% supervise
  er_all(:, c) = er;
  wij_all(:, :, c) = wij_wi;
  % wjt_all(:, :, c) = wjt;
  % wtl_all(:, :, c) = wtl;
end

figure(1)
% subplot(2, 2, 1);plot(1:Nbmp, sum(er_all.^2), 'r-');
% subplot(2, 2, 2);plot(1:Nbmp, reshape(sum(sum(wij_all.^2)), Nbmp, 1), 'b-');ylabel('wij');
% subplot(2, 2, 3);plot(1:Nbmp, reshape(sum(sum(wjt_all.^2)), Nbmp, 1), 'c-');ylabel('wjt');
% subplot(2, 2, 4);plot(1:Nbmp, reshape(sum(sum(wtl_all.^2)), Nbmp, 1), 'd-');ylabel('wtl');

figure(2)
for f2 = 1:10;
  subplot(2, 5, f2); imshow(reshape(wij_wi(:,f2 * nh1 / 10), 28, 28) .* 200, [-300 300]);
end




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
    % hl_t = sigmoid(sigmoid((vt)' * wij_wi./nin) * wjt./nh1) * wtl;
    hl_t = sigmoid(sigmoid((vt)' * wij_wi) * wjt) * wtl;
    [y, p] = max(hl_t);
    confusion(digit_t + 1, p) = confusion(digit_t + 1, p) + 1;
  end
end

accuracy = 100*sum(diag(confusion))/sum(sum(confusion));
fprintf('Total Accuracy = %2.1f%%\n',accuracy);




% accuracy = zeros(1,10);
%         confusion = zeros(10,10);
%         Fin_val = zeros(100,10);
%         for CL = 1:10
%             for c = 800:999;
%                 dist = zeros(1,10);
%                 fnamet = sprintf('digit_%1d_%03d.bmp', CL-1, c);
%                 % A = double(imread(['/Users/timer/OneDrive/ms1_2/neuralnetwork/hw5/test_data_new/' fnamet]))./255;
%                 B = double(imread(['~/OneDrive/ms1_2/neuralnetwork/hw5/test_data_new/' fnamet]))./255;
%                 dist = (reshape(B,784,1)'*wij_wi*wjt*wtl);
%                 [y,indmin] = max(dist);
%                 confusion(CL,indmin) = confusion(CL,indmin)+1;
%                 Fin_val(c-699,CL) = y;
%             end
%         end
% accuracy = 100*sum(diag(confusion))/sum(sum(confusion));
% fprintf('Total Accuracy = %2.1f%%\n',accuracy);
