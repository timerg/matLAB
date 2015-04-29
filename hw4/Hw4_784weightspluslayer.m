


clear all; close all;

Ntrain = 699; %800 pictures seperate into two parts
Ntest = 99;
W = 28; %28*28 picso
wj = roundn(rand(W^2,20)*0.01,-0);
wk = roundn(rand(20,10),-1);
% wj_all = zeros(W,10)
Amean = ones(W^2,10);
etak = 0.01;
etaj = 0.01;
%% read images and calculate the mean image for each digit
for ll = 0:Ntrain*20
    for k = 0;
        for n = 1:1;
            
%         n = ceil(rand*10)-1
%         c = ceil(rand*0.7*1000)-1
            d = zeros(1,10);
            d(n+1) = 0.5

            for c = [0:(k*100-1),(k*100+100):(Ntrain+100)];
                fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',n,c);
                B = double(imread(fname));
                xj = reshape(B./255,784,1);

                yj = xj'*wj./784;%wj':1x20
                xk = 1./(1.+exp(-yj));  %1x20
                yk = xk*wk./20;% 1x10;
                err = d-yk

                %%% change all
                
                for nn = 0:9;
                delta = err(nn+1)*wk(:,nn+1).*xk'.*(1.-xk');
                wj = wj + etaj.*xj*delta';

                %%% change once a time

                wk(:,nn+1) = wk(:,nn+1) + etak*err(:,nn+1).*xk';
                end
            end
        end
    end
end
        for n = 0:9; 
            ww = wj*wk;
            Amean(:,n+1) = reshape(ww(:,n+1).*255*100,W^2,1);
            figure(1); subplot(3,4,n+1); 
            imshow(reshape(Amean(:,n+1),W,W),[0 255]) 
        end
%% testing
accuracy = zeros(1,10);
confusion = zeros(10,10);
Fin_val = zeros(100,10)
for CL = 1:10
    for c = 700:799
        dist = zeros(1,10);
        fnamet = sprintf('digit_%1d_%03d.bmp', CL-1, c);
        A = double(imread(['/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/' fnamet]))./255;
        dist = (reshape(A,784,1)'*wj*wk);
        [y,indmin] = max(dist);
        confusion(CL,indmin) = confusion(CL,indmin)+1;
        Fin_val(c-699,CL) = y;
    end
end
fprintf('Total Accuracy = %2.1f%%\n',100*sum(diag(confusion))/sum(sum(confusion)));

