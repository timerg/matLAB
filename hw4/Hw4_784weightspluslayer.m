


clear all; close all;

Ntrain = 699; %800 pictures seperate into two parts
Ntest = 99;
W = 28; %28*28 picso
wi = ones(W^2,1);
wj = ones(W^2,20).*0.2;
wk = ones(20,10).*0.5;
% wj_all = zeros(W,10)
Amean = ones(W^2,10);
eta = 0.001;
%% read images and calculate the mean image for each digit
for k = 0:0;
for n = 0:9;
    d = zeros(10,1);
%      for c1 = Ntrain:799;
%         fname1 = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',n,c1);
%         A = imread(fname1);
%         Ad = double(A);
%         d = d+(sum(Ad)./W./255)';
%         
%      end
%         dforb = sum(d./(28*100));
    d(n+1) = 100;
    
    for c = [0:(k*100-1),(k*100+100):(Ntrain+100)];
        fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',n,c);
        B = double(imread(fname));
        xj = reshape(B./255,784,1);
%         d = d + (sum(B))';

        
        yj = xj'*wj;%wj':1x20
        xk = 1./(1.+exp(-yj));  %1x20
        yk = xk*wk;% 1x10;
        err = d'-yk
%         err = d-yj';  % every one
%         delta = (err.*xj.*(1.-xj)'*wj(:,n+1)
%         delta =(xj.*(1.-xj))'*wj*err;   %every one
%         wi = wi +eta*delta.*xi;
%         wj = wj + eta*xj*err';   %every one
%         wk(:,n+1) = wk(:,n+1) + eta*err.*xk';
       
%         delta = err.*xk.*(1-xk).*wk(:,n+1)';
        delta = err*wk'.*xk.*(1-xk);
        wj = wj+eta.*xj*delta;
         wk = wk+eta.*xk'*err;
%         yj(n+1) = 0;
%         [va,n2] = max(yj);
%         errn = 0.5-yj(n2)';
%         wj(:,n2) = wj(:,n2) + eta*errn.*xj;

    end
%       Amean(:,n+1) = reshape(wj(:,n+1).*255*100,W^2,1);
%     figure(1); subplot(3,4,n+1); 
%     imshow(reshape(Amean(:,n+1),W,W),[0 255])  
%     
    

end
    



%% testing
accuracy = zeros(1,10);
confusion = zeros(10,10);
for CL = 1:10
    for c = k*100:k*100+99
        dist = zeros(1,10);
        fname = sprintf('digit_%1d_%03d.bmp', CL-1, c);
        A = double(imread(['/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/' fname]))./255;
        dist = (reshape(A,784,1)'*wj*wk);
        [y,indmin] = max(dist);
        confusion(CL,indmin) = confusion(CL,indmin)+1;
    end
end
fprintf('Total Accuracy = %2.1f%%\n',100*sum(diag(confusion))/sum(sum(confusion)));

end