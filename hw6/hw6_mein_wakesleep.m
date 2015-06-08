clear all;
close all;

% hj = randi([0 1], 200, 1);
% hk = randi([0 1], 50, 1);
%% weight
wij_r = rand(784, 200);
wij_w = rand(200, 784);
wij_r = rand(200, 50);
wjk_w = rand(50, 200);

%% bias
b1 = 0;
b2 = 0;

%% parameters
eta = 0.05;
Nbmp = 1000

% mode
gibbs = 0;
only2 = 1;
% train
for c = 1:Nbmp;
    cc = ceil(rand*1000);
    if only2 == 1,
        % fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',cc-1);
        fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',cc);   %for windows
    elseif only2 == 0,
        % fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',floor(rand*10),cc-1);
        fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',floor(rand*10),cc);
    end
    A = double(imread(fname));
    vi = reshape(A./255, 784, 1);
    ((vi)' * wij_r)' = hj;         %200x1
    ((hj)'' * wjk_r)' = hk;        %50x1
    for aa = 1:1;
      aa = aa;
    % Total energy

      E1  = (hk)' * wjk_w;           %1x200
      p1 = 1 ./ (1 + exp(-b1 - E1));        %200x1
      wjk_w = wjk_w + hk * ((hj)' - p1)

      E2  = (hj)' * wij_w;           %1x200
      p2 = 1 ./ (1 + exp(-b2 - E2));        %200x1
      wij_w = wij_w + hj * ((vi)' - p2)

      E3  = (vi)' * wij;
      p3 = 1 ./ (1 + exp(-E3));
      Etot3 = vi * p3;
      t3 = rand(1, 50) .* 0.6 + 0.2;
      % t3 = 0.5;
      hj = (p1> t3)';

      E4  = (hj)' * (wij)';      %E2 is 1x784
      p4 = 1 ./ (1 + exp(-E4));
      t4 = rand(1, 784) .* 0.6 + 0.2;
      % t4 = 0.5;
      vi = (p4> t4)';
    % change wij
      wij = wij+eta*(Etot1-Etot3);
    end


  if mod(c,Nbmp) == 0,
    figure(1);
    for cp = 1:50;
          pic = round(reshape(((wij(:,cp)+1).*(255/2)),28,28));
          subplot(5,10,cp); imshow( pic,[min(min(pic)) max(max(pic))]); hold on;
    end
  end
  if rem(c,(Nbmp/10)) == 0,
      figure(2)
      subplot(3, 10, c/Nbmp * 10); imshow(A, [0 255]);
      subplot(3, 10, 10 + c / Nbmp * 10); imshow(reshape((vi .* 255), 28, 28), [0 255]);
      if ~gibbs,
      subplot(3,10 , 20 + c / Nbmp * 10); imshow(reshape((p4 .* 255), 28, 28), [0 255]);
      end
  end
end
   c=c
