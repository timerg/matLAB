


clear all; close all;

Ntrain = 599; %800 pictures seperate into two parts
Ntest = 99;
W = 28; %28*28 picso
wi = ones(W^2,1);
wj = ones(W^2,10).*0.5;
% wj_all = zeros(W,10)
Amean = ones(W^2,10);
eta = 0.0001;
%% read images and calculate the mean image for each digit
for k = 0:1000;
% for n = 0:9;
    n = ceil(rand*10)-1;
    c = ceil(rand*0.7*1000)-1;
    d = zeros(10,1);
%      for c1 = Ntrain:799;
%         fname1 = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',n,c1);
%         A = imread(fname1);
%         Ad = double(A);
%         d = d+(sum(Ad)./W./255)';
%
%      end
%         dforb = sum(d./(28*100));
    d(n+1) = 1

%     for c = [0:(k*100-1),(k*100+100):(Ntrain+100)];
        % fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',n,c);
        fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',n,c);
        B = double(imread(fname));
        xi = reshape(B./255,784,1);
%         d = d + (sum(B))';
        yi = xi.*wi;   %784x1
        xj = 1./(1.+exp(yi));
        yj = xj'*wj;%wj':1x10
        err = d(n+1)-yj(n+1)';
%         err = d-yj';  % every one
%         delta = (err.*xj.*(1.-xj)'*wj(:,n+1)
%         delta =(xj.*(1.-xj))'*wj*err;   %every one
%         wi = wi +eta*delta.*xi;
%         wj = wj + eta*xj*err';   %every one
        wj(:,n+1) = wj(:,n+1) + eta*err.*xj;
%         yj(n+1) = 0;
%         [va,n2] = max(yj);
%         errn = 0.5-yj(n2)';
%         wj(:,n2) = wj(:,n2) + eta*errn.*xj;

end
    for n = 0:9;
      Amean(:,n+1) = reshape(wj(:,n+1).*255*100,W^2,1);
    figure(1); subplot(3,4,n+1);
    imshow(reshape(Amean(:,n+1),W,W),[0 255])
    end


% end

% end


%% testing
accuracy = zeros(1,10);
confusion = zeros(10,10);
for CL = 1:10
    for c = 700:799
        dist = zeros(1,10);
        fname = sprintf('digit_%1d_%03d.bmp', CL-1, c);
        % A = double(imread(['/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/' fname]));
        A = double(imread(['~/OneDrive/ms1_2/neuralnetwork/hw4/data/' fname]));
        dist = (reshape(A,784,1).*wi)'*wj;
        [y,indmin] = max(dist);
        confusion(CL,indmin) = confusion(CL,indmin)+1;
    end
end
fprintf('Total Accuracy = %2.1f%%\n',100*sum(diag(confusion))/sum(sum(confusion)));
