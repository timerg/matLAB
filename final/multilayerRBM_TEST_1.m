clear all;
close all;




nh1 = 500;
nh2 = 700;
nt = 2000;
nd = 10;
nin = 784;
nbph = 100;

%% weight
wij_w = roundn((rand(nin, nh1) - 0.5), -3);
wjk_w = roundn((rand(nh1, nh2) - 0.5), -3);
% Top RBM weoghts
wkt = rand(nh2, nt) - 0.5;
% bp
wi = rand(nt, nbph);
wj = rand(nbph, nd);
% simple
sw = rand(nt, nd*3);
%% bias
b0 = 0;
b1 = 0;
b2 = 0;
b3 = 0;
b4 = 0;
%% parameters
etaa = 0.01;
etat = 0.1;
etai = 0.01;
etaj = 0.01;
etas = 0.1;
Nbmp = 200;
tt = 0.5;

% mode
only2 = 0;
rr = 1;
% train
for c = 1:Nbmp
  c
    cc = ceil(rand*999);
    if only2 == 1,
        digit = 2;
        % fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',cc-1);
        fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',cc);   %for windows
    elseif only2 == 0,
        digit = floor(rand*10);
        % fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',digit,cc-1);
        fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',floor(rand*10),cc);
    end
    A = double(imread(fname));
    vi = reshape(A./255, 784, 1);
    vi_origin = vi;

    for aa = 1: 1
%%% wij
        %% 1st froward
                Ejf = (vi_origin)' * wij_w;
                pijf = 1 ./ (1 + exp(-Ejf));
                hj = gt(pijf, rand(1, nh1))';
                Etotij = vi * pijf;

                Ejr  = (hj)' * (wij_w)';           %
                pjir = 1 ./ (1 + exp(-Ejr));        %
                vi_1 = gt(pjir, rand(1, nin))';
        %% 2nd forward
                Ejf = (vi_1)' * wij_w;
                pijf = 1 ./ (1 + exp(-Ejf));
                hj = gt(pijf, rand(1, nh1))';
                Etotij2 = vi_1 * pijf;

        wij_w = wij_w + etaa .* (Etotij - Etotij2);
% %% weight jk
%         %% 1st froward
%                 Ejf = (vi_origin)' * wij_w;
%                 pijf = 1 ./ (1 + exp(-Ejf));
%                 hj = gt(pijf, rand(1, nh1))';
%
%                 Ekf = (hj)' * wjk_w;
%                 pjkf = 1 ./ (1 + exp(-Ekf));
%                 hk = gt(pjkf, rand(1, nh2))';   % 500x1
%                 Etotjk = hj * pjkf;
%
%                 Etf  = (hk)' * wkt;       %1x2000
%                 pktf = 1 ./ (1 + exp(-Etf));
%                 ht = gt(pktf, rand(1, nt))';
%         %% 1st backward
%                 Etr  = (ht)' * (wkt)';
%                 ptkr = 1 ./ (1 + exp(-Etr));
%                 hk = gt(ptkr, rand(1, nh2))';
%
%                 Ekr  = (hk)' * (wjk_w)';           %
%                 pkjr = 1 ./ (1 + exp(-Ekr));        %
%                 hj = gt(pkjr, rand(1, nh1))';
%
%                 Ejr  = (hj)' * (wij_w)';           %
%                 pjir = 1 ./ (1 + exp(-Ejr));        %
%                 vi_2 = gt(pjir, rand(1, nin))';
%         %% 2nd forward
%                 Ejf = (vi_2)' * wij_w;
%                 pijf = 1 ./ (1 + exp(-Ejf));
%                 hj = gt(pijf, rand(1, nh1))';
%
%                 Ekf = (hj)' * wjk_w;
%                 pjkf = 1 ./ (1 + exp(-Ekf));
%                 hk = gt(pjkf, rand(1, nh2))';   % 500x1
%                 Etotjk2 = hj * pjkf;
%
%         wjk_w = wjk_w + etaa .* (Etotjk - Etotjk2);
% %% weight kt
%         %% 1st froward
%                 Ejf = (vi_origin)' * wij_w;
%                 pijf = 1 ./ (1 + exp(-Ejf));
%                 hj = gt(pijf, rand(1, nh1))';
%
%                 Ekf = (hj)' * wjk_w;
%                 pjkf = 1 ./ (1 + exp(-Ekf));
%                 hk = gt(pjkf, rand(1, nh2))';   % 500x1
%
%                 Etf  = (hk)' * wkt;       %1x2000
%                 pktf = 1 ./ (1 + exp(-Etf));
%                 ht = gt(pktf, rand(1, nt))';
%                 Etotkt = hk * pktf;
%         %% 1st backward
%                 Etr  = (ht)' * (wkt)';
%                 ptkr = 1 ./ (1 + exp(-Etr));
%                 hk = gt(ptkr, rand(1, nh2))';
%
%                 Ekr  = (hk)' * (wjk_w)';           %
%                 pkjr = 1 ./ (1 + exp(-Ekr));        %
%                 hj = gt(pkjr, rand(1, nh1))';
%
%                 Ejr  = (hj)' * (wij_w)';           %
%                 pjir = 1 ./ (1 + exp(-Ejr));        %
%                 vi_3 = gt(pjir, rand(1, nin))';
%         %% 2nd forward
%                 Ejf = (vi_3)' * wij_w;
%                 pijf = 1 ./ (1 + exp(-Ejf));
%                 hj = gt(pijf, rand(1, nh1))';
%
%                 Ekf = (hj)' * wjk_w;
%                 pjkf = 1 ./ (1 + exp(-Ekf));
%                 hk = gt(pjkf, rand(1, nh2))';   % 500x1
%
%                 Etf  = (hk)' * wkt;       %1x2000
%                 pktf = 1 ./ (1 + exp(-Etf));
%                 ht = gt(pktf, rand(1, nt))';
%                 Etotkt2 = hk * pktf;
%
%         wkt = wkt + etat .*(Etotkt - Etotkt2);
%
%  %% simple
%          Ejf = (vi_origin)' * wij_w;
%          pijf = 1 ./ (1 + exp(-Ejf));
%          hj = gt(pijf, rand(1, nh1))';
%
%          Ekf = (hj)' * wjk_w;
%          pjkf = 1 ./ (1 + exp(-Ekf));
%          hk = gt(pjkf, rand(1, nh2))';   % 500x1
%
%          Etf  = (hk)' * wkt;       %1x2000
%          pktf = 1 ./ (1 + exp(-Etf));
%          ht = gt(pktf, rand(1, nt))';
%         for ss = 1:10
%           sh = (ht)' * sw ./ nt;
%           sdi = zeros(1,nd*3);
%           sdi(digit * 3 +1: digit * 3 +1 +2) = 10;
%           ser = sdi - sh;
%           sdelta = ht * ser;
%           sw = sw + etas .* sdelta;
%         end
%  %% bp
%         bpdi = zeros(1, 10);
%         bpdi(digit + 1) = 1;
%         bpv = (ht)';       % 1x2000
%         bph = bpv * wi ./nbph;    % 1x50
%         bphs = sigmoid(bph);
%         bpy = bphs * wj ./ nd;   %1x10
%         bper = bpdi - bpy;
%         bpdelta_i = (bpv)' * (wj * bper' .* dsigmoid(bphs'))';
%         bpdelta_j = bphs' * bper;
%
%         wi = wi + bpdelta_i .* etai;
%         wj = wj + bpdelta_j .* etaj;

        if rem(c,(Nbmp/10)) == 0,
            figure(2);title('during train')
            subplot(4, 10, c/Nbmp * 10); imshow(reshape((vi_origin .* 255), 28, 28), [0 255]);
            subplot(4, 10, 10 + c / Nbmp * 10); imshow(reshape((vi_1 .* 255), 28, 28), [0 255]);
            % subplot(4, 10, 20 + c / Nbmp * 10); imshow(reshape((vi_2 .* 255), 28, 28), [0 255]);
            % subplot(4, 10, 30 + c / Nbmp * 10); imshow(reshape((vi_3 .* 255), 28, 28), [0 255]);
            % subplot(3, 10, 20 + c / Nbmp * 10); imshow(reshape((wij_w(1,:) .* 255), 28, 28), [0 255]);
        end
    end
end


%% test
t_times = 500;
I = zeros(1, t_times);
testresult = zeros(10, 10);
testresult_s = 0;
for tc = 1:t_times;
  tcc = floor(rand * 999);
  if only2,
    digit_t = 2;
  else
    digit_t = floor(rand*10);
  end
  Iin(1, tc) = digit_t + 1;   %recording
  % ftname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp', digit_t, tcc);
  ftname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp', digit_t, tcc);
  B = double(imread(ftname));
  vt = reshape(B./255, 784, 1);

  Etij = (vt)' * wij_w;
      pvtij = 1 ./ (1 + exp(-Etij));
      vtj = gt(pvtij, rand(1, nh1))';
  Etji = (vtj)' * (wij_w)';
      pvtji = 1 ./ (1 + exp(-Etji));
      vtr = gt(pvtji, rand(1, nin))';

  % Etkj2 = (vtk)' * (wjk_w)';
  %     pvtkj2 = 1 ./ (1 + exp(-Etkj2));
  %     vtj_r2 = gt(pvtkj2, rand(1, nh1))';
  % Etji2 = (vtj_r)' * (wij_w)';
  %     pvtji2 = 1 ./ (1 + exp(-Etji2));
  %     vtr2 = gt(pvtji2, rand(1, nin))';
  %
  % Etji3 = (vtj)' * (wij_w)';
  %     pvtji3 = 1 ./ (1 + exp(-Etji3));
  %     vtr3 = gt(pvtji3, rand(1, nin))';

%% bp
%   [by bp] = max(1./(1+ exp(-(((htt)' * wi )* wj))));
%   testresult(bp, (digit_t+1)) = testresult(bp, (digit_t + 1)) + 1;
%   accuracy_bp = sum(diag(testresult))/sum(sum(testresult)) * 100;
% %% simple
%   [sy sp] = max((htt)' * sw ./ nt);
%   testresult_s = testresult_s + max([((digit_t + 1)== sp) ((digit_t + 2)== sp) ((digit_t + 3)== sp)]);
%   accuracy_s = testresult_s/t_times * 100;

  % if out(digit_t + 1, 1) == 1,
  %   testresult(1, tc) = 1;
  % end
  % [M, I(1, tc)] = max(out);
  if mod(tc, (t_times / 10)) == 0,
    figure(3)
    subplot(4, 10, tc / (t_times / 10)); imshow(B, [0 255]);
    subplot(4, 10, 10 + tc / (t_times / 10)); imshow(reshape((vtr .* 255), 28, 28), [0 255]);
    % subplot(4, 10, 20 + tc / (t_times / 10)); imshow(reshape((vtr2 .* 255), 28, 28), [0 255]);
    % subplot(4, 10, 30 + tc / (t_times / 10)); imshow(reshape((vtr3 .* 255), 28, 28), [0 255]);
  end
end
fprintf('accuracy_bp = %2.1f%%\n', accuracy_bp)
fprintf('accuracy_s = %2.1f%%\n', accuracy_s)
%
% fprintf('accuracy = %2.1f%%\n', 100 * sum(testresult) / t_times)
%
% figure(4)
% subplot(4,1,1);plot((1:nh2), p1, 'r-');
% subplot(4,1,2);plot((1:784), p2, 'r-');
% subplot(4,1,3);plot((1:nh2), p3, 'r-');
% subplot(4,1,4);plot((1:nh2), p4, 'r-');
%
% figure(5)
% plot(1:t_times, I, 'g-', 1:t_times, Iin, 'r-');
%
%
% wkt = wktl(1:nh2, :);
% wtl = (wktl((nh2 + 1):(nh2 + nd), :))';
% wij_wi = (wij_w)';
% wjk_wi = (wjk_w)';

%% save data
% save('~/GitHub/matLAB/final/weight_ij', 'wij_wi')
% save('~/GitHub/matLAB/final/weight_jk', 'wjk_wi')
% save('~/GitHub/matLAB/final/weight_kt', 'wkt')
% save('~/GitHub/matLAB/final/weight_tl', 'wtl')
% save('~/GitHub/matLAB/final/hidden_j', 'hj')
% save('~/GitHub/matLAB/final/hidden_k', 'hk')
% save('~/GitHub/matLAB/final/hidden_t', 'ht')
% save('~/GitHub/matLAB/final/hidden_l', 'hl')

% save('/Users/timer/Documents/GitHub/matLAB/final/weight_ij', 'wij_wi')
% save('/Users/timer/Documents/GitHub/matLAB/final/weight_jk', 'wjk_wi')
% save('/Users/timer/Documents/GitHub/matLAB/final/weight_kt', 'wkt')
% save('/Users/timer/Documents/GitHub/matLAB/final/weight_tl', 'wtl')
% save('/Users/timer/Documents/GitHub/matLAB/final/hidden_j', 'hj')
% save('/Users/timer/Documents/GitHub/matLAB/final/hidden_k', 'hk')
% save('/Users/timer/Documents/GitHub/matLAB/final/hidden_t', 'ht')
% save('/Users/timer/Documents/GitHub/matLAB/final/hidden_l', 'hl')
