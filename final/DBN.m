clear all;
close all;




nh1 = 500;
nh2 = 500;
nt = 2000;
nd = 10;

%% weight
wij_r = rand(784, nh1) - 0.5;
wij_w = rand(nh1, 784) - 0.5;
wjk_r = rand(nh1, nh2) - 0.5;
wjk_w = rand(nh2, nh1) - 0.5;
% Top RBM weoghts
wktl = rand(nh2 + nd, nt) - 0.5;

%% bias
b0 = 0;
b1 = 0;
b2 = 0;
b3 = 0;
b4 = 0;
%% parameters
etaa = 0.01;
Nbmp = 100000;
tt = 0.5;

% mode
gibbs = 0;
only2 = 0;
rr = 1;
% train
for c = 1:Nbmp
    cc = ceil(rand*999);
    if only2 == 1,
        digit = 2;
%         fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',cc-1);
        fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',cc);   %for windows
    elseif only2 == 0,
        digit = floor(rand*10);
%         fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',digit,cc-1);
        fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',floor(rand*10),cc);
    end
    A = double(imread(fname));
    vi = reshape(A./255, 784, 1);

    for aa = 1: 1
% wake phase
        Ej = ((vi)' * wij_r)';
        if rr,
          hj = gt((1 ./ (1 + exp(-Ej))), rand);
        else
          hj = gt((1 ./ (1 + exp(-Ej))), tt);         %200x1
        end
        Ek = ((hj)' * wjk_r)';
        if rr,
          hk = gt((1 ./ (1 + exp(-Ek))), rand);
        else
          hk = gt((1 ./ (1 + exp(-Ek))), tt);         %50x1
        end

        %%
        E_w1  = (hk)' * wjk_w;           %1x200
        p1 = 1 ./ (1 + exp(-b1 - E_w1));        %200x1
        wjk_w = wjk_w + etaa .* hk * ((hj)' - p1);

        E_w2  = (hj)' * wij_w;           %1x200
        p2 = 1 ./ (1 + exp(-b2 - E_w2));        %200x1
        wij_w = wij_w + etaa .* hj * ((vi)' - p2);
  % sleep phase
        Ej = ((hk)' * wjk_w)';
        if rr,
          hj = gt((1 ./ (1 + exp(-Ej))), rand);
        else
          hj = gt((1 ./ (1 + exp(-Ej))), tt);
        end
        Ei = ((hj)' * wij_w)';
        if rr,
          vi = gt((1 ./ (1 + exp(-Ei))), rand);
        else
          vi = gt((1 ./ (1 + exp(-Ei))), tt);
        end

        E_s1  = (vi)' * wij_r;
        p3 = 1 ./ (1 + exp(-b3-E_s1));
        wij_r = wij_r + etaa .* vi * ((hj)' - p3);

        E_s2  = (hj)' * wjk_r;      %E2 is 1x784
        p4 = 1 ./ (1 + exp(-E_s2));
        wjk_r = wjk_r + etaa .* hj * ((hk)' - p4);

        %% Top RBM with label train
                hl = zeros(10, 1);
                hl(digit + 1) = 1;
                hkl = [hk;hl];      %510x1
                Ekt1  = (hkl)' * wktl;       %1x2000
                pkt1 = 1 ./ (1 + exp(-Ekt1));
                Etot1 = hkl * pkt1;
                ht = gt(pkt1, rand)';       %2000x1
          % end
                Etk1  = (ht)' * (wktl)';
                ptk1 = 1 ./ (1 + exp(-Etk1));
                hkl = gt(ptk1, rand)';

                Ekt2  = (hkl)' * wktl;
                pkt2 = 1 ./ (1 + exp(-Ekt2));
                Etot2 = hkl * pkt2;
                ht = gt(pkt2, rand)';

                Etk2  = (ht)' * (wktl)';      %E2 is 1x784
                ptk2 = 1 ./ (1 + exp(-Etk2));
                hkl = gt(ptk2, rand)';
          % change wij
                wktl = wktl + etaa*(Etot1-Etot2);

  end

%% to show the reconstruction
Er1 = ((vi)' * (wij_w)')';
pr1 = sigmoid(Er1);
hrj = gt(pr1, rand);
Er2 =


  if rem(c,(Nbmp/10)) == 0,
      figure(2)
      subplot(3, 10, c/Nbmp * 10); imshow(A, [0 255]);
      subplot(3, 10, 10 + c / Nbmp * 10); imshow(reshape((vi .* 255), 28, 28), [0 255]);
      subplot(3, 10, 20 + c / Nbmp * 10); imshow(reshape((wij_w(1,:) .* 255), 28, 28), [0 255]);
      if ~gibbs,
      % subplot(3,10 , 20 + c / Nbmp * 10); imshow(reshape((p4 .* 255), 28, 28), [0 255]);
      end
  end
end

%% test
t_times = 500;
I = zeros(1, t_times);
testresult = zeros(1, t_times);
for tc = 1:t_times;
  tcc = floor(rand * 999);
  if only2,
    digit_t = 2;
  else
    digit_t = floor(rand*10);
  end
  Iin(1, tc) = digit_t + 1;   %recording
%   ftname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp', digit_t, tcc);
  ftname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp', digit_t, tcc);
  B = double(imread(ftname));
  vt = reshape(B./255, 784, 1);
  pvtij = 1 ./ (1 + exp(-(vt)' * (wij_w)'));
  vtj = gt(pvtij, rand)';
  pvtjk = 1 ./ (1 + exp(-(vtj)'  * (wjk_w)'));
  vtk = gt(pvtjk, rand)';

  Et1 =  [(vtk)' randi([-1, 1],1 ,nd)] * wktl;
  pt1 = 1 ./ (1 + exp(-Et1));
  htt = gt(pt1, rand)';
  pt2 = 1 ./ (1 + exp(-(htt)'  * (wktl)'));
  vtk_final = gt(pt2, rand)';

  pvtkj = 1 ./ (1 + exp(-(vtk)' * wjk_w));
  vtj2 = gt(pvtkj, rand)';
  pvtji = 1 ./ (1 + exp(-(vtj2)' * wij_w));
  vtr = gt(pvtji, rand)';

  out = vtk_final((nh2 + 1):(nh2 + nd));
  if out(digit_t + 1, 1) == 1,
    testresult(1, tc) = 1;
  end
  [M, I(1, tc)] = max(out);
  if mod(tc, (t_times / 10)) == 0,
    figure(3)
    subplot(2, 10, tc / (t_times / 10)); imshow(B, [0 255]);
    subplot(2, 10, 10 + tc / (t_times / 10)); imshow(reshape((vtr .* 255), 28, 28), [0 255]);
  end
end

fprintf('accuracy = %2.1f%%\n', 100 * sum(testresult) / t_times)

figure(4)
subplot(4,1,1);plot((1:nh2), p1, 'r-');
subplot(4,1,2);plot((1:784), p2, 'r-');
subplot(4,1,3);plot((1:nh2), p3, 'r-');
subplot(4,1,4);plot((1:nh2), p4, 'r-');

figure(5)
plot(1:t_times, I, 'g-', 1:t_times, Iin, 'r-');


wkt = wktl(1:nh2, :);
wtl = (wktl((nh2 + 1):(nh2 + nd), :))';
wij_wi = (wij_w)';
wjk_wi = (wjk_w)';

%% save data
% save('~/GitHub/matLAB/final/weight_ij', 'wij_wi')
% save('~/GitHub/matLAB/final/weight_jk', 'wjk_wi')
% save('~/GitHub/matLAB/final/weight_kt', 'wkt')
% save('~/GitHub/matLAB/final/weight_tl', 'wtl')
% save('~/GitHub/matLAB/final/hidden_j', 'hj')
% save('~/GitHub/matLAB/final/hidden_k', 'hk')
% save('~/GitHub/matLAB/final/hidden_t', 'ht')
% save('~/GitHub/matLAB/final/hidden_l', 'hl')

save('/Users/timer/Documents/GitHub/matLAB/final/weight_ij', 'wij_wi')
save('/Users/timer/Documents/GitHub/matLAB/final/weight_jk', 'wjk_wi')
save('/Users/timer/Documents/GitHub/matLAB/final/weight_kt', 'wkt')
save('/Users/timer/Documents/GitHub/matLAB/final/weight_tl', 'wtl')
save('/Users/timer/Documents/GitHub/matLAB/final/hidden_j', 'hj')
save('/Users/timer/Documents/GitHub/matLAB/final/hidden_k', 'hk')
save('/Users/timer/Documents/GitHub/matLAB/final/hidden_t', 'ht')
save('/Users/timer/Documents/GitHub/matLAB/final/hidden_l', 'hl')
