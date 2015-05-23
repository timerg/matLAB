clear all;
close all;

hj = zeros(50,1);
wij = (round(rand(784,50).*100)-50).*(2/5) ;

for c = 0:0;
% fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',c);
fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',c);   %for windows
A = double(imread(fname));
vi = reshape(A./255,784,1);

En  = (vi)'*wij.*(hj)';      %En is 1x50
En_ = (vi)'*wij.*(abs(hj-1))';
gap = En - En_
  for cc = 1:50;
    if gap(1,cc) > 0;
      hj(cc,1) = hj(cc,1);
  elseif gap(1,cc) < 0;
      hj(cc,1) = abs(hj(cc,1)-1);
    end
  end

end
