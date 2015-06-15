clear all;
close all;





nh1 = 200;
nh2 = 50;
% function
% function y = sigmoid(x);
%   y = 1 ./(1 + exp^(-x))
% end
%% weight
% (wij_w)' = rand(784, nh1) - 0.5;
wij_w = rand(nh1, 784) - 0.5;
% (wjk_w)' = rand(nh1, nh2) - 0.5;
wjk_w = rand(nh2, nh1) - 0.5;

%% bias
b0 = 0;
b1 = 0;
b2 = 0;
b3 = 0;
b4 = 0;
%% parameters
etaa = 0.1;
Nbmp = 1;
tt = 0.5;

% mode
gibbs = 0;
only2 = 1;
rr = 0;
% train
for c = 1:Nbmp;
    cc = ceil(rand*999);
    if only2 == 1,
        % fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',cc-1);
        fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',cc);   %for windows
    elseif only2 == 0,
        % fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',floor(rand*10),cc-1);
        fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',floor(rand*10),cc);
    end
    A = double(imread(fname));
    vi = reshape(A./255, 784, 1);

    for aa = 1: 100
% wake phase
      Ej = ((vi)' * (wij_w)')';
      if rr,
        hj = gt((1 ./ (1 + exp(-Ej))), rand);
      else
        hj = gt((1 ./ (1 + exp(-Ej))), tt);         %200x1
      end
      Ek = ((hj)' * (wjk_w)')';
      if rr,
        hk = gt((1 ./ (1 + exp(-Ek))), rand);
      else
        hk = gt((1 ./ (1 + exp(-Ek))), tt);         %50x1
      end

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

      E_s1  = (vi)' * (wij_w)';
      p3 = 1 ./ (1 + exp(-b3 - E_s1));
    %   (wij_w)' = (wij_w)' + etaa .* vi * ((hj)' - p3);
      wij_w = ((wij_w)' + etaa .* vi * ((hj)' - p3))';

      E_s2  = (hj)' * (wjk_w)';      %E2 is 1x784
      p4 = 1 ./ (1 + exp(-E_s2));
    %   (wjk_w)' = (wjk_w)' + etaa .* hj * ((hk)' - p4);
      wjk_w = ((wjk_w)' + etaa .* hj * ((hk)' - p4))';

  end


  if mod(c,Nbmp) == 0,
    figure(1);
    for cp = 1:50;
          % pic = round(reshape(((wij_w(:,cp)+1).*(255/2)),28,28));
          % subplot(5,10,cp); imshow( pic,[min(min(pic)) max(max(pic))]); hold on;
    end
  end
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
%% text
t_times = 10;
for tc = 1:t_times;
  tcc = floor(rand * tc * 100);
  % ftname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',floor(rand*10), tcc);
  ftname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',floor(rand*10), tcc);
  B = double(imread(ftname));
  vt = reshape(B./255, 784, 1);
  pvtij = 1 ./ (1 + exp(-(vt)' * ((wij_w)')));
  vtj = gt(pvtij, rand);
  pvtjk = 1 ./ (1 + exp(-vtj  * ((wjk_w)')));
  vtk = gt(pvtjk, rand);
  pvtkj = 1 ./ (1 + exp(-vtk * wjk_w));
  vtj2 = gt(pvtkj, rand);
  pvtji = 1 ./ (1 + exp(-vtj2 * wij_w));
  vtr = gt(pvtji, rand);
  figure(3)
  subplot(3, 10, tc); imshow(B, [0 255]);
  subplot(3, 10, 10 + tc); imshow(reshape((vtr .* 255), 28, 28), [0 255]);
end

figure(4)
subplot(4,1,1);plot((1:200), p1, 'r-'); ylabel('p1');
subplot(4,1,2);plot((1:784), p2, 'r-'); ylabel('p2');
subplot(4,1,3);plot((1:200), p3, 'r-'); ylabel('p3');
subplot(4,1,4);plot((1:50), p4, 'r-'); ylabel('p4');
