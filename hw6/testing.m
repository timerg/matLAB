


clear all; close all
load('~/GitHub/matlab/hw6/hw6_weight.mat');
for r = 1:10;
  fnametest = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp', floor(rand*10), floor(rand*999));
  T = double(imread(fnametest));
  vt0 = reshape(T ./ 255, 784, 1);

  Et1  = (vt0)' * wij;
  pt1 = 1 ./ (1 + exp(-Et1));
  ht = (pt1 > 0.5)';

  Et2  = (ht)' * (wij)';
  pt2 = 1 ./ (1 + exp(-Et2));
  vt = (pt2> 0.5)';
  figure(5);
  subplot(2,10,r); imshow(reshape((vt0 .* 255),28,28),[0 255]);
  subplot(2,10,r+10); imshow(reshape((vt .* 255),28,28),[0 255]);
end
