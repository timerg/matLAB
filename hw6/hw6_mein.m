clear all;
close all;

hj = randi([0 1],50,1);
wij = ((rand(784,50)));
eta = 0.01;
for c = 0:100;
% fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',c);
fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',c);   %for windows
A = double(imread(fname));
vi = reshape(A./255,784,1);
  for aa = 1:1000;
      aa=aa
% Total energy
% two ways
      % E1  = (vi)'*wij.*(hj)';      %En is 1x50
      % E1_ = (vi)'*(wij).*(abs(hj-1))';
      % gap = E1 - E1_;
      % for cc = 1:50;
      %     if gap(1,cc) > 0;
      %         hj(cc,1) = hj(cc,1);
      %   elseif gap(1,cc) < 0;
      %         hj(cc,1) = abs(hj(cc,1)-1);
      %     end
      % end
% or
      E1  = (vi)'*wij;      %probability for hj = 1
      p1 = 1./(1+exp(-E1));        %％also, for hj(x)=0, E1(x)=0, p1(x)=0.5
      Etot1 = vi*p1;              %the expectation value is: p*vin*h(=1)   p is still come from h,v
      hj = (p1> 0.5)';
% end
      E2  = (hj)'*(wij)';      %E2 is 1x784
      p2 = 1./(1+exp(-E2));
      vi = (p2> 0.5)';

      E3  = (vi)'*wij;
      p3 = 1./(1+exp(-E3));
      Etot3 = vi*p3;
      hj = (p1> 0.5)';

      E4  = (hj)'*(wij)';      %E2 is 1x784
      p4 = 1./(1+exp(-E4));
      vi = (p4> 0.5)';

% change wij
      wij = wij+eta*(Etot1-Etot3);
    end
  figure(1);
  for cc = 1:50;
    pic = round(reshape(((wij(:,cc)+1).*(300/2)),28,28));
    subplot(5,10,cc); imshow( pic,[0 255]); hold on;
  end
  c=c
end
