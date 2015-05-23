clear all;
close all;

hj = randi([0 1],50,1);
wij = (round(rand(784,50).*40)-20);
eta = 0.1;
for c = 0:100;
  c=c
fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',c);
% fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',c);   %for windows
A = double(imread(fname));
vi = reshape(A./255,784,1);
  for aa = 1:100
      aa=aa
% Total energy
      Etot = vi*(hj)'.*wij;
      E1  = (vi)'*wij.*(hj)';      %En is 1x50
      E1_ = (vi)'*wij.*(abs(hj-1))';
      gap = E1 - E1_;
      for cc = 1:50;
          if gap(1,cc) > 0;
              hj(cc,1) = hj(cc,1);
        elseif gap(1,cc) < 0;
              hj(cc,1) = abs(hj(cc,1)-1);
          end
      end
      E2  = (hj)'*(wij)'.*(vi)';      %En is 1x50
      E2_ = (hj)'*(wij)'.*(abs(vi-1))';
      gap = E2 - E2_;
      for cc = 1:784;
          if gap(1,cc) > 0;
              vi(cc,1) = vi(cc,1);
        elseif gap(1,cc) < 0;
              vi(cc,1) = abs(vi(cc,1)-1);
          end
      end
% change wij
      Etot_ = vi*(hj)'.*wij;
      wij = wij+eta*(Etot-Etot_);
    end
  % figure(1)
  for cc = 1:10;
    pic = (reshape(wij(:,cc),28,28)+20).*255/40;
    subplot(5,10,cc); imshow( pic,[0 255]); hold on;
  end
end
