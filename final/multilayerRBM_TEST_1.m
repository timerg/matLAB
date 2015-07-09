clear all;
close all;




nh1 = 20;
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
b1 = -5;
b2 = 0;
b3 = 0;
b4 = 0;

%% parameters
etaa = 0.1;
etat = 0.1;
etai = 0.01;
etaj = 0.01;
etas = 0.1;
Nbmp = 1000;
tt = 0.5;
picnum = 82;
% mode
only2 = 0;
rr = 1;
% train
for c = 1:Nbmp
  c
  cc = ceil(rand*picnum);
  % fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/final/H%01d.bmp',cc);
  fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/final/H%01d.bmp',cc);
  A = double(imread(fname));
  vi = reshape(A./255, nin, 1);
    for aa = 1: 1
        dev = std(vi);
        bi = mean(vi);
        bi_a = mean(vi) - 0.5;
        bi_b = bi_a - 0.5;
%%% wij
        %% 1st froward
        % Ejf_a = sum((vi - bi_a).^2 ./ 2 /dev^2) - (vi)' * wij_w ./ dev;
        % Ejf_b = sum((vi - bi_b).^2 ./ 2 /dev^2) - (vi)' * wij_w ./ dev;
        % Ejf_c = sum((vi - bi_c).^2 ./ 2 /dev^2) - (vi)' * wij_w ./ dev;
        % Ejf_d = sum((vi - bi_d).^2 ./ 2 /dev^2) - (vi)' * wij_w ./ dev;
        % Ejf_e = sum((vi - bi_e).^2 ./ 2 /dev^2) - (vi)' * wij_w ./ dev;
        % Ejf_f = sum((vi - bi_f).^2 ./ 2 /dev^2) - (vi)' * wij_w ./ dev;
        % Ejf = sum([Ejf_a;Ejf_b;Ejf_c;Ejf_d;Ejf_e;Ejf_f])
        % Ejf = (vi)' * wij_w + randi([0, ceil(dev)], 1, nh1) + bi;
        Ejf = sum((vi - bi).^2 ./ 2/ (dev^2)) - (vi)' * wij_w ./ dev;
                pijf = 1 ./ (1 + exp(-Ejf));
                hj = gt(pijf, rand(1, nh1))';
                % hj = max(0, -mean(Ejf) + Ejf + randi([0, ceil(dev)]))';
                Etotij = vi * pijf;

                % Ejr  = log(1+ exp((hj)' * (wij_w)'));
                Ejr =  (hj)' * (wij_w)';
                pjir = 1 ./ (1 + exp(-Ejr));        %
                vi_1 = (pjir)';
        %% 2nd forward
        dev = std(vi_1);
        bi = mean(vi_1);
        % Ejf2_a = sum((vi_1 - bi_a).^2 ./ 2 /dev^2) - (vi_1)' * wij_w ./ dev;
        % Ejf2_b = sum((vi_1 - bi_b).^2 ./ 2 /dev^2) - (vi_1)' * wij_w ./ dev;
        % Ejf2_c = sum((vi_1 - bi_c).^2 ./ 2 /dev^2) - (vi_1)' * wij_w ./ dev;
        % Ejf2_d = sum((vi_1 - bi_d).^2 ./ 2 /dev^2) - (vi_1)' * wij_w ./ dev;
        % Ejf2_e = sum((vi_1 - bi_e).^2 ./ 2 /dev^2) - (vi_1)' * wij_w ./ dev;
        % Ejf2_f = sum((vi_1 - bi_f).^2 ./ 2 /dev^2) - (vi_1)' * wij_w ./ dev;
        % Ejf2 = sum([Ejf2_a;Ejf2_b;Ejf2_c;Ejf2_d;Ejf2_e;Ejf2_f]);
        % Ejf2 = (vi_1)' * wij_w + randi([0, ceil(dev)], 1, nh1) + bi;
        Ejf2 = sum((vi_1 - bi).^2 ./ 2 ./ dev^2) - (vi_1)' * wij_w ./ dev;
                % Ejf = (vi_1)' * wij_w;
                pijf2 = 1 ./ (1 + exp(-Ejf2));
                hj2 = gt(pijf2, rand(1, nh1))';
                Etotij2 = vi_1 * pijf2;
        wij_w = wij_w + etaa .* (Etotij - Etotij2);



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
  % ftname = sprintf('~/OneDrive/ms1_2/neuralnetwork/final/T%1d.bmp' , tc);
  ftname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/final/T%1d.bmp' , tc);
  B = double(imread(ftname));
  tv = reshape(B./255, nin, 1);
  tdev = std(tv);
  tbi = -mean(tv);
  tEij = (tv)' * wij_w + randi([0, ceil(tdev)], 1, nh1) + tbi;
  % Ejf = (vi_origin)' * wij_w;
  tpij = 1 ./ (1 + exp(-tEij));
  thj = gt(tpij, rand(1, nh1))';
  % thj = max(0, tEij)';

  tEji  = tbi + (thj)' * (wij_w)';           %
  tpji = 1 ./ (1 + exp(-tEji));        %
  tvi_1 = (tpji)';


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
    subplot(4, 10, tc / (testnum / 10)); imshow(B, [0 255]);
    subplot(2, 10, 10 + tc / (testnum / 10)); imshow(reshape((tvi_1 .* 255), sqrt(nin), sqrt(nin)), [0 255]);
  end
end
figure(4)
for cp = 1:50;
      pic = round(reshape(((wij_w(:,cp)+1).*(255/2)),32,32));
      subplot(5,10,cp); imshow( pic,[min(min(pic)) max(max(pic))]); hold on;
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
