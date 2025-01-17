


clear all; close all;

Ntrain = 700; %800 pictures seperate into two parts
Ntest = 100;
W = 28; %28*28 picso
Amean = zeros(W^2,10); % mean images
wj = (ones(1,W))';
wi = (rand(W,W))';
% wj_all = zeros(W,10)
d = (zeros(1,28))';
eta = 0.001;
%% read images and calculate the mean image for each digit
for n = 0:9;

     for c1 = Ntrain:799;
        fname1 = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',n,c1);
        A = imread(fname1);
        Ad = double(A);
        d = d+(sum(Ad)./W./255)';
        
     end
        dforb = sum(d./(28*100));
    
    
    for c = 1:Ntrain
        fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',n,c);
        B = imread(fname);
        xi = double(B)./255;
%         d = d + (sum(B))';
        yi = (sum(xi.*wi)./28)';    %xi,wi: 28x28 yi': 1x28
        xj = 1./(1.+exp(yi));
        yj = xj'*wj;%wj':1x28
        err = dforb-yj;
        delta =repmat ((err.*wj.*xj.*(1.-xj))',28,1);
        wi = wi +eta*delta.*xi;
        wj = wj + eta.*err.*xj;
        

    end
    wj_all(:,n+1) = wj;
    

%     Amean(:,n+1) = reshape(mu/Ntrain,W^2,1);
    figure(1); subplot(3,4,n+1); 
     imshow(wi.*255,[0 255])
end
    


%% testing
% B = zeros(W,W,10);
% for n = 0:9
%     B(:,:,n+1) = reshape(Amean(:,n+1),W,W);
% end
% accuracy = zeros(1,10);
% confusion = zeros(10,10);
% for CL_fact = 1:10
%     for c = 1:Ntest
%         dist = zeros(1,10);
%         fname = sprintf('digit_%1d_%03d.bmp', CL_fact-1, c+Ntrain);
%         A = double(imread([DIR fname]));
%         for CL = 1:10
%             dist(CL) = sum(sum((A-B(:,:,CL)).^2));
%         end
%         [y,indmin] = min(dist);
%         confusion(CL_fact,indmin) = confusion(CL_fact,indmin)+1;
%     end
% end
% fprintf('Total Accuracy = %2.1f%%\n',100*sum(diag(confusion))/sum(sum(confusion)));