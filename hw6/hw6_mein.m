clear all;
close all;

hj = randi([0 1],50,1);
wij = ((rand(784,50)))-0.5;
eta = 0.05;
for c = 0:999;
% fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',c);
fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',c);   %for windows
A = double(imread(fname));
vi = reshape(A./255,784,1);
  for aa = 1:1;
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
    %   t1 = rand(1,50);
      t1 = 0.5;
      hj = (p1> t1)';
% end
      E2  = (hj)'*(wij)';      %E2 is 1x784
      p2 = 1./(1+exp(-E2));
    %   t2 = rand(1,784);
      t2 = 0.5;
      vi = (p2> t2)';

      E3  = (vi)'*wij;
      p3 = 1./(1+exp(-E3));
      Etot3 = vi*p3;
    %   t3 = rand(1,50);
      t3 = 0.5;
      hj = (p1> t3)';

      E4  = (hj)'*(wij)';      %E2 is 1x784
      p4 = 1./(1+exp(-E4));
    %   t4 = rand(1,784);
      t4 = 0.5;
      vi = (p4> t4)';
% change wij
      wij = wij+eta*(Etot1-Etot3);
  end


  if mod(c+1,1000) ==0,
    figure(1);
    for cc = 1:50;
          pic = round(reshape(((wij(:,cc)+1).*(255/2)),28,28));
          subplot(5,10,cc); imshow( pic,[56 190]); hold on;
    end
  end
   c=c
end
figure(2);
imshow(reshape((vi.*255),28,28),[0 255])
