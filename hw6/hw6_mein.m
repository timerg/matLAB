clear all;
close all;

hj = randi([0 1],50,1);
wij = ((rand(784,50)))-0.5;
eta = 0.05;
Nbmp = 2000

% mode
gibbs = 0;
only2=1;
% train
for c = 1:Nbmp;
    cc = ceil(rand*1000);
    if only2 == 1,
        fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',cc-1);
        % fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',cc);   %for windows
    elseif only2 == 0,
        fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',floor(rand*10),cc-1);
        % fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',floor(rand*10),cc);
    end
A = double(imread(fname));
vi = reshape(A./255, 784, 1);


  if ~gibbs,
    for aa = 1:1;
      aa = aa;
% Total energy

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
  elseif gibbs,
    Etotr = zeros(1000,1);
    Ntrain = 10000
    for aa = 1:Ntrain;
      vh = ceil(rand*50);
      hj0 = hj;
      hj1 = hj;
      hj0(vh, :) = 0;
      hj1(vh, :) = 1;
      Eoff = ((vi)' * wij * hj0);
      Eon = ((vi)' * wij * hj1);
      deltaE1 = Eoff-Eon;
      p1 = 1/(1 + exp(-deltaE1));
      Etot1 = p1 .* vi;
      hj(vh,:) = (p1 > 0.5);

      hv = ceil(rand * 784);
      vi0 = vi;
      vi1 = vi;
      vi0(hv, :) = 0;
      vi1(hv, :) = 1;
      Eoff = (hj)' * (wij)'*vi0;
      Eon = (hj)'*(wij)'*vi1;
      deltaE2 = Eoff-Eon;
      p2 = 1/(1+exp(-deltaE2));
      vi(hv,:) = (p2>0.5);

      hj0 = hj;
      hj1 = hj;
      hj0(vh,:) = 0;
      hj1(vh,:) = 1;
      Eoff = (vi)'*wij*hj0;
      Eon = (vi)'*wij*hj1;
      deltaE3 = Eoff-Eon;
      p3= 1/(1+exp(-deltaE3));
      if aa == Ntrain,
          Etot3 = p3.*vi;
      end
      hj(vh,:) = (p3>0.5);

      vi0 = vi;
      vi1 = vi;
      vi0(hv,:) = 0;
      vi1(hv,:) = 1;
      Eoff = (hj)'*(wij)'*vi0;
      Eon = (hj)'*(wij)'*vi1;
      deltaE4 = Eoff-Eon;
      p4 = 1/(1+exp(-deltaE4));
      vi(hv,:) = (p4>0.5);
      if aa == Ntrain,
          wij(:,vh) = wij(:,vh)-eta*(Etot1-Etot3);
      end
      Etotr(aa,1) = deltaE4;
    end
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
   c=c
end
% figure(4)
%   plot([1:Ntrain],Etotr,'r-')
