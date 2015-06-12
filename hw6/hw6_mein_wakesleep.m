clear all;
close all;

%% weight
wij_r = randi([-1,1],784, 200);
wij_w = randi([-1,1],200, 784);
wjk_r = randi([-1,1],200, 50);
wjk_w = randi([-1,1],50, 200);

%% bias
b0 = 0;
b1 = 0;
b2 = 0;
b3 = 0;
b4 = 0;
%% parameters
etaa = 0.05;
Nbmp = 1000

% mode
gibbs = 0;
only2 = 1;
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
      Ej = ((vi)' * wij_r)';
      hj = gt((1 ./ (1 + exp(-Ej))), 0.5);         %200x1
      Ek = ((hj)' * wjk_r)';
      hk = gt((1 ./ (1 + exp(-Ek))), 0.5);         %50x1

      E_w1  = (hk)' * wjk_w ./ 50;           %1x200
      p1 = 1 ./ (1 + exp(-b1 - E_w1));        %200x1
      wjk_w = wjk_w + etaa .* hk * ((hj)' - p1);

      E_w2  = (hj)' * wij_w ./ 200;           %1x200
      p2 = 1 ./ (1 + exp(-b2 - E_w2));        %200x1
      wij_w = wij_w + etaa .* hj * ((vi)' - p2);
% sleep phase
      Ej = ((hk)' * wjk_w)';
      hj = gt((1 ./ (1 + exp(-Ej))), 0.5);
      Ei = ((hj)' * wij_w)';
      vi = gt((1 ./ (1 + exp(-Ei))), 0.5);

      E_s1  = (vi)' * wij_r ./ 784;
      p3 = 1 ./ (1 + exp(-b3-E_s1));
      wij_r = wij_r + etaa .* vi * ((hj)' - p3);

      E_s2  = (hj)' * wjk_r ./200;      %E2 is 1x784
      p4 = 1 ./ (1 + exp(-E_s2));
      wjk_r = wjk_r + etaa .* hj * ((hk)' - p4);

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
   c=c
