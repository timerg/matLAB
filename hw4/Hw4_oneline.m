


clear all; close all;

Ntrain = 599; %800 pictures seperate into two parts
Ntest = 99;
W = 28; %28*28 picso
wi = zeros(W,10);
wj = zeros(W,10);

eta = 0.001;
%% read images and calculate the mean image for each digit
for k = 0:0;
    for n = 0:9;
        d = zeros(10,1);  
        d(n+1) =1 
        for c = [0:(k*100-1),(k*100+100):(Ntrain+101)];
            fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',n,c);
            B = imread(fname);
            xi = (sum(double(B)./(255*28)))';
            yi = (xi'*wi)';
            err = 1-yi(n+1);
            wi(:,n+1) = wi(:,n+1)+eta*xi*err;     
        end

        for r = [0:(k*100-1),(k*100+100):(Ntrain+101)];
           fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',n,r);
            B = imread(fname);
            xi = (sum(double(B)'./(255*28)))';
            yj = (xi'*wj)';
            err = 1-yj(n+1);
            wj(:,n+1) = wj(:,n+1)+eta*xi*err;      
        end
    end
end




%% testing
C = zeros(W,W,10);
for n = 0:9
end
accuracy = zeros(1,10);
confusion = zeros(10,10);
for CL_fact = 1:10
    for c = 1:Ntest
        dist = zeros(1,10);
        distr = zeros(1,10);
        distc = zeros(1,10);
        fname = sprintf('digit_%1d_%03d.bmp', CL_fact-1, c+Ntrain);
        A = double(imread(['/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/' fname]));
        distc =sum(A)*wi;
        distr = sum(A')*wj;
        dist = distc.*distr;
        [y,indmin] = max(dist);
        confusion(CL_fact,indmin) = confusion(CL_fact,indmin)+1;
    end
end
fprintf('Total Accuracy = %2.1f%%\n',100*sum(diag(confusion))/sum(sum(confusion)));
%% testing 2
% C = zeros(W,W,10);
% for n = 0:9
% end
% confusion2 = zeros(10,10);
% for CL_fact = 1:10
%     for c = 1:Ntest
%         dist2 = zeros(1,10);
%         fname = sprintf('digit_%1d_%03d.bmp', CL_fact-1, c+Ntrain);
%         A = double(imread(['/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/' fname]));
%         for tt = 1:10
%             dist2(:,tt) =sum(sum((A*diag(wi(:,tt)))'*diag(wj(:,tt))));
%         end
%         
%         [y,indmin2] = max(dist2);
%         confusion2(CL_fact,indmin) = confusion2(CL_fact,indmin)+1;
%     end
% end
% fprintf('Total Accuracy = %2.1f%%\n',100*sum(diag(confusion2))/sum(sum(confusion2)));