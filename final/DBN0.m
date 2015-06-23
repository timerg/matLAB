clear all;
close all;




nh1 = 500;
nh2 = 500;
nt = 2000;
nd = 10;
nin = 784;
nbph = 100;

%% weight
wij_r = rand(nin, nh1) - 0.5;
wij_w = rand(nh1, nin) - 0.5;
wjk_r = rand(nh1, nh2) - 0.5;
wjk_w = rand(nh2, nh1) - 0.5;
% Top RBM weoghts
wkt = rand(nh2, nt) - 0.5;
% bp
wi = rand(nt, nbph);
wj = rand(nbph, nd);
% simple
sw = rand(nt, nd*3);
%% bias
b0 = 0.1;
b1 = 0.1;
b2 = 0.1;
b3 = 0.1;
b4 = 0.1;
%% parameters
etaa = 0.01;
etat = 0.01;
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
        fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',cc-1);
        % fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',cc);   %for windows
    elseif only2 == 0,
        digit = floor(rand*10);
        fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',digit,cc-1);
        % fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',floor(rand*10),cc);
    end
    A = double(imread(fname));
    vi = reshape(A./255, 784, 1);

    for aa = 1: 1
% wake phase
        Ej = ((vi)' * wij_r)';
        hj = gt((1 ./ (1 + exp(-Ej))), rand(nh1, 1));
        Ek = ((hj)' * wjk_r)';
        hk = gt((1 ./ (1 + exp(-Ek))), rand(nh2, 1));

        %%
        E_w1  = (hk)' * wjk_w;           %1x200
        p1 = 1 ./ (1 + exp(-b1 - E_w1));        %200x1
        wjk_w = wjk_w + etaa .* hk * ((hj)' - p1);

        E_w2  = (hj)' * wij_w;           %1x200
        p2 = 1 ./ (1 + exp(-b2 - E_w2));        %200x1
        wij_w = wij_w + etaa .* hj * ((vi)' - p2);
  % sleep phase
        Ej = ((hk)' * wjk_w)';
        hj = gt((1 ./ (1 + exp(-Ej))), rand(nh2, 1));
        Ei = ((hj)' * wij_w)';
        vi = gt((1 ./ (1 + exp(-Ei))), rand(nin, 1));


        E_s1  = (vi)' * wij_r;
        p3 = 1 ./ (1 + exp(-b3-E_s1));
        wij_r = wij_r + etaa .* vi * ((hj)' - p3);

        E_s2  = (hj)' * wjk_r;      %E2 is 1x784
        p4 = 1 ./ (1 + exp(-E_s2));
        wjk_r = wjk_r + etaa .* hj * ((hk)' - p4);
        if rem(c,(Nbmp/10)) == 0,
            figure(2);title('during train')
            subplot(2, 10, c/Nbmp * 10); imshow(A, [0 255]);
            subplot(2, 10, 10 + c / Nbmp * 10); imshow(reshape((vi .* 255), 28, 28), [0 255]);
            % subplot(3, 10, 20 + c / Nbmp * 10); imshow(reshape((wij_w(1,:) .* 255), 28, 28), [0 255]);
        end
    end
end


%% Top RBM with label train
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
    Ej = ((vi)' * (wij_w)')';
    hj = gt((1 ./ (1 + exp(-Ej))), rand(nh1, 1));
    Ek = ((hj)' * (wjk_w)')';
    hk = gt((1 ./ (1 + exp(-Ek))), rand(nh2, 1));
    Ekt1  = (hk)' * wkt;       %1x2000
    pkt1 = 1 ./ (1 + exp(-Ekt1));
    Etot1 = hk * pkt1;
    ht = gt(pkt1, rand(1, nt))';       %2000x1
% end
    Etk1  = (ht)' * (wkt)';
    ptk1 = 1 ./ (1 + exp(-Etk1));
    hk = gt(ptk1, rand(1, nh2))';

    Ekt2  = (hk)' * wkt;
    pkt2 = 1 ./ (1 + exp(-Ekt2));
    Etot2 = hk * pkt2;
    ht = gt(pkt2, rand(1, nt))';    % will be used to train bp

    Etk2  = (ht)' * (wkt)';      %E2 is 1x784
    ptk2 = 1 ./ (1 + exp(-Etk2));
    hk = gt(ptk2, rand(1, nh2))';
% change wij
    wkt = wkt + etat*(Etot1-Etot2);
  %% simple
    for ss = 1:10
      sh = (ht)' * sw ./ nt;
      sdi = zeros(1,nd*3);
      sdi(digit * 3 +1: digit * 3 +1 +2) = 10;
      ser = sdi - sh;
      sdelta = ht * ser;
      sw = sw + etas .* sdelta;
    end
  %% bp
    bpdi = zeros(1, 10);
    bpdi(digit + 1) = 1;
    bpv = (ht)';       % 1x2000
    bph = bpv * wi ./nbph;    % 1x50
    bphs = sigmoid(bph);
    bpy = bphs * wj ./ nd;   %1x10
    bper = bpdi - bpy;
    bpdelta_i = (bpv)' * (wj * bper' .* dsigmoid(bphs'))';
    bpdelta_j = bphs' * bper;

    wi = wi + bpdelta_i .* etai;
    wj = wj + bpdelta_j .* etaj;
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
  pvtij = 1 ./ (1 + exp(-(vt)' * (wij_w)'));
  vtj = gt(pvtij, rand(1, nh1))';
  pvtjk = 1 ./ (1 + exp(-(vtj)'  * (wjk_w)'));
  vtk = gt(pvtjk, rand(1, nh2))';

  Et1 =  (vtk)' * wkt;
  pt1 = 1 ./ (1 + exp(-Et1));
  htt = gt(pt1, rand(1, nt))';
  pt2 = 1 ./ (1 + exp(-(htt)'  * (wkt)'));
  vtk_r = gt(pt2, rand(1, nh2))';

  pvtkj = 1 ./ (1 + exp(-(vtk_r)' * wjk_w));
  vtj2 = gt(pvtkj, rand(1, nh1))';
  pvtji = 1 ./ (1 + exp(-(vtj2)' * wij_w));
  vtr = gt(pvtji, rand(1, nin))';

%% bp
  [by bp] = max(1./(1+ exp(-(((htt)' * wi )* wj))));
  testresult(bp, (digit_t+1)) = testresult(bp, (digit_t + 1)) + 1;
  accuracy_bp = sum(diag(testresult))/sum(sum(testresult)) * 100;
%% simple
  [sy sp] = max((htt)' * sw ./ nt);
  testresult_s = testresult_s + max([((digit_t + 1)== sp) ((digit_t + 2)== sp) ((digit_t + 3)== sp)]);
  accuracy_s = testresult_s/t_times * 100;

  % if out(digit_t + 1, 1) == 1,
  %   testresult(1, tc) = 1;
  % end
  % [M, I(1, tc)] = max(out);
  if mod(tc, (t_times / 10)) == 0,
    figure(3)
    subplot(2, 10, tc / (t_times / 10)); imshow(B, [0 255]);
    subplot(2, 10, 10 + tc / (t_times / 10)); imshow(reshape((vtr .* 255), 28, 28), [0 255]);
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
