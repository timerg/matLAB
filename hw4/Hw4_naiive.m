% Mean-image method for na?ve hand-written digit recognition
% April 7, 2015
% YWLiu
clear; close all;
% DIR = '~/Desktop/NN/DigitFiles/';
DIR = '/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/';
Ntrain = 700; %800 pictures seperate into two parts
Ntest = 100;
W = 28; %28*28 picso
Amean = zeros(W^2,10); % mean images

%% read images and calculate the mean image for each digit
for n = 0:9
    mu = zeros(W,W);
    for c = 1:Ntrain
        fname = sprintf('digit_%1d_%03d.bmp',n,c);
        A = imread([DIR fname]);
        mu = mu+double(A);  %about double: http://stackoverflow.com/questions/801117/whats-the-difference-between-a-single-precision-and-double-precision-floating-p
    end
    Amean(:,n+1) = reshape(mu/Ntrain,W^2,1);
    figure(1); subplot(3,4,n+1); 
    imshow(reshape(Amean(:,n+1),W,W),[0 255])
    
end

%% testing
B = zeros(W,W,10);
for n = 0:9
    B(:,:,n+1) = reshape(Amean(:,n+1),W,W);
end
accuracy = zeros(1,10);
confusion = zeros(10,10);
for CL_fact = 1:10
    for c = 1:Ntest
        dist = zeros(1,10);
        fname = sprintf('digit_%1d_%03d.bmp', CL_fact-1, c+Ntrain);
        A = double(imread([DIR fname]));
        for CL = 1:10
            dist(CL) = sum(sum((A-B(:,:,CL)).^2));   %cal the difference on each class by sum(e^2)
        end
        [y,indmin] = min(dist)  %find the one has the least difference.  y is vlaue; indmin is position
        confusion(CL_fact,indmin) = confusion(CL_fact,indmin)+1  %if correct, confusion should only have value(=1) on diagnos
    end
end
fprintf('Total Accuracy = %2.1f%%\n',100*sum(diag(confusion))/sum(sum(confusion))); %diag(confusion) will put all diag elements into one column vector


