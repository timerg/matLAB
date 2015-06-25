clear all;
close all;




nh1 = 500;
nh2 = 700;
nt = 2000;
nd = 10;
nin = 1024;
nbph = 100;

%% weight
wij_w = ceil((rand(nin, nh1) - 0.5) .* 1000)./1000;
wjk_w = ceil((rand(nh1, nh2) - 0.5) .* 1000)./1000;
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
etaa = 1;
etat = 1;
etai = 0.01;
etaj = 0.01;
etas = 0.1;
Nbmp = 100;
tt = 0.5;
picnum = 82;
% mode
only2 = 0;
rr = 1;
% train
for c = 1:Nbmp
  c
  cc = ceil(rand*picnum);
  fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/final/H%01d.bmp',cc);
  % fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/final/H%01d.bmp',cc);
  A = double(imread(fname));
  vi = reshape(A./255, nin, 1);
    for aa = 1: 1
        dev = std(vi);
        bi_a = mean(vi) - 0.5;
        bi_b = bi_a - 0.5
%%% wij
        %% 1st froward
        % Ejf_a = sum((vi - bi_a).^2 ./ 2 /dev^2) - (vi)' * wij_w ./ dev;
        % Ejf_b = sum((vi - bi_b).^2 ./ 2 /dev^2) - (vi)' * wij_w ./ dev;
        % Ejf_c = sum((vi - bi_c).^2 ./ 2 /dev^2) - (vi)' * wij_w ./ dev;
        % Ejf_d = sum((vi - bi_d).^2 ./ 2 /dev^2) - (vi)' * wij_w ./ dev;
        % Ejf_e = sum((vi - bi_e).^2 ./ 2 /dev^2) - (vi)' * wij_w ./ dev;
        % Ejf_f = sum((vi - bi_f).^2 ./ 2 /dev^2) - (vi)' * wij_w ./ dev;
        % Ejf = sum([Ejf_a;Ejf_b;Ejf_c;Ejf_d;Ejf_e;Ejf_f])
        Ejf = (vi)' * wij_w./ dev + bi_a
                pijf = 1 ./ (1 + exp(-Ejf));
                hj = gt(pijf, rand(1, nh1))';
                Etotij = vi * pijf;

                Ejr  = (hj)' * (wij_w)';           %
                pjir = 1 ./ (1 + exp(-Ejr));        %
                vi_1 = (pjir)';
        %% 2nd forward
        dev = std(vi_1);
        bi = mean(vi_1);
        Ejf2_a = sum((vi_1 - bi_a).^2 ./ 2 /dev^2) - (vi_1)' * wij_w ./ dev;
        Ejf2_b = sum((vi_1 - bi_b).^2 ./ 2 /dev^2) - (vi_1)' * wij_w ./ dev;
        Ejf2_c = sum((vi_1 - bi_c).^2 ./ 2 /dev^2) - (vi_1)' * wij_w ./ dev;
        Ejf2_d = sum((vi_1 - bi_d).^2 ./ 2 /dev^2) - (vi_1)' * wij_w ./ dev;
        Ejf2_e = sum((vi_1 - bi_e).^2 ./ 2 /dev^2) - (vi_1)' * wij_w ./ dev;
        Ejf2_f = sum((vi_1 - bi_f).^2 ./ 2 /dev^2) - (vi_1)' * wij_w ./ dev;
        Ejf2 = sum([Ejf2_a;Ejf2_b;Ejf2_c;Ejf2_d;Ejf2_e;Ejf2_f]);
                % Ejf = (vi_1)' * wij_w;
                pijf2 = 1 ./ (1 + exp(-Ejf2));
                hj2 = gt(pijf2, rand(1, nh1))';
                Etotij2 = vi_1 * pijf2;
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
            subplot(4, 10, c/Nbmp * 10); imshow(reshape((vi .* 255), sqrt(nin), sqrt(nin)), [0 255]);
            subplot(4, 10, 10 + c / Nbmp * 10); imshow(reshape((vi_1 .* 255), sqrt(nin), sqrt(nin)), [0 255]);
            % subplot(4, 10, 20 + c / Nbmp * 10); imshow(reshape((vi_2 .* 255), sqrt(nin), sqrt(nin)), [0 255]);
            % subplot(4, 10, 30 + c / Nbmp * 10); imshow(reshape((vi_3 .* 255), sqrt(nin), sqrt(nin)), [0 255]);
            % subplot(3, 10, 20 + c / Nbmp * 10); imshow(reshape((wij_w(1,:) .* 255), sqrt(nin), sqrt(nin)), [0 255]);
        end
    end
end


%% test
% testresult_s = 0;
testnum = 10;
for tc = 1:testnum;
  ftname = sprintf('~/OneDrive/ms1_2/neuralnetwork/final/T%1d.bmp' , tc);
  % ftname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/final/T%1d.bmp' , tc);
  B = double(imread(ftname));
  vtt = reshape(B./255, nin, 1);
  devt = std(vtt);
  bit = mean(vtt);
  Ejf = sum((vtt - bi).^2 ./ dev^2) - (vtt)' * wij_w ./ dev;
  % Ejf = (vi_origin)' * wij_w;
  pijf = 1 ./ (1 + exp(-Ejf));
  hj = gt(pijf, rand(1, nh1))';
  Etotij = vtt * pijf;

  Ejr  = (hj)' * (wij_w)';           %
  pjir = 1 ./ (1 + exp(-Ejr));        %
  vti_1 = (pjir)';


%
% % %% simple
% %   [sy sp] = max((htt)' * sw ./ nt);
% %   testresult_s = testresult_s + max([((digit_t + 1)== sp) ((digit_t + 2)== sp) ((digit_t + 3)== sp)]);
% %   accuracy_s = testresult_s/t_times * 100;
%
%   % if out(digit_t + 1, 1) == 1,
%   %   testresult(1, tc) = 1;
%   % end
%   % [M, I(1, tc)] = max(out);
  if mod(tc, (testnum / 10)) == 0,
    figure(3)
    subplot(2, 10, tc / (testnum / 10)); imshow(B, [0 255]);
    subplot(2, 10, 10 + tc / (testnum / 10)); imshow(reshape((vti_1 .* 255), sqrt(nin), sqrt(nin)), [0 255]);
  end
end
% fprintf('accuracy_bp = %2.1f%%\n', accuracy_bp)
% fprintf('accuracy_s = %2.1f%%\n', accuracy_s)
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
