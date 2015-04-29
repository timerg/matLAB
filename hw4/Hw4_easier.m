


clear all; close all;

Ntrain = 599; %800 pictures seperate into two parts
Ntest = 99;
W = 28; %28*28 picso
sp = 2
wi = zeros(W,10);
wj = zeros(W,10);

eta = 0.01;
%% read images and calculate the mean image for each digit
for k = 0:7840;
%     for n = 0:9;
        n = ceil(rand*10)-1
        c = ceil(rand*0.7*1000)-1
        d = zeros(10,1);  
        d(n+1) =1 
%         for c = [0:(k*100-1),(k*100+100):(Ntrain+101)];
            fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',n,c);
            B = double(imread(fname));
            xic = (sum(B)./(255*W))';
            xir = (sum(B')./(255*W))';
            yi = (xic'*wi)';
            yj = (xir'*wj)';
%             errc =d(n+1)'-yi(n+1);
%             errr =d(n+1)'-yj(n+1);
%             wi(:,n+1) = wi(:,n+1)+eta.*xic.*errc;
%             wj(:,n+1) = wj(:,n+1)+eta.*xir.*errr;   
            errc =d'-yi';
            errr =d'-yj';
            wi = wi+eta.*xic*errc;
            wj = wj+eta.*xir*errr;   
            
        end
%     end
% end




%% testing
C = zeros(W,W,10);
for n = 0:9
end
accuracy = zeros(1,10);
confusion = zeros(10,10);
for CL_fact = 1:10
    for c = 1:Ntest
        dist = zeros(1,10);
        distr = zeros(4,10);
        distc = zeros(4,10);
        fname = sprintf('digit_%1d_%03d.bmp', CL_fact-1, c+Ntrain);
        A = double(imread(['/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/' fname]))./255;
        Ac = sum(A);
        distc = Ac*wi
        Ar = sum(A');
        distr = Ar*wj;
        dist = distc+distr;
        [y,indmin] = max(dist);
        confusion(CL_fact,indmin) = confusion(CL_fact,indmin)+1;
    end
end
fprintf('Total Accuracy = %2.1f%%\n',100*sum(diag(confusion))/sum(sum(confusion)));