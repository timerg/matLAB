


clear all; close all;

Ntrain = 599; %800 pictures seperate into two parts
Ntest = 99;
W = 28; %28*28 picso
sp = 2
wi = zeros(W/sp,10,sp^2);
wj = zeros(W/sp,10,sp^2);

eta = 0.001;
%% read images and calculate the mean image for each digit
for k = 0:0;
    for n = 0:9;
        d = zeros(10,1);  
        d(n+1) =1 
        for c = [0:(k*100-1),(k*100+100):(Ntrain+101)];
            fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',n,c);
            B = double(imread(fname));
            B_1 = B(1:W/sp,1:W/sp);
            B_2 = B(1:W/sp,W/sp+1:W);
            B_3 = B(W/sp+1:W,1:W/sp);
            B_4 = B(W/sp+1:W,W/sp+1:W);
            xic_1 = (sum(B_1)./(255*W/sp))';
            xic_2 = (sum(B_2)./(255*W/sp))';
            xic_3 = (sum(B_3)./(255*W/sp))';
            xic_4 = (sum(B_4)./(255*W/sp))';
            xir_1 = (sum(B_1')./(255*W/sp))';
            xir_2 = (sum(B_2')./(255*W/sp))';
            xir_3 = (sum(B_3')./(255*W/sp))';
            xir_4 = (sum(B_4')./(255*W/sp))';
            yi_1 = (xic_1'*wi(:,:,1))';
            yi_2 = (xic_2'*wi(:,:,2))';
            yi_3 = (xic_3'*wi(:,:,3))';
            yi_4 = (xic_4'*wi(:,:,4))';
            yj_1 = (xir_1'*wj(:,:,1))';
            yj_2 = (xir_2'*wj(:,:,2))';
            yj_3 = (xir_3'*wj(:,:,3))';
            yj_4 = (xir_4'*wj(:,:,4))';
            errc_1 = 1-yi_1(n+1);
            errc_2 = 1-yi_2(n+1);
            errc_3 = 1-yi_3(n+1);
            errc_4 = 1-yi_4(n+1);
            errr_1 = 1-yj_1(n+1);
            errr_2 = 1-yj_2(n+1);
            errr_3 = 1-yj_3(n+1);
            errr_4 = 1-yj_4(n+1);
            wi(:,n+1,1) = wi(:,n+1,1)+eta*xic_1*errc_1;
            wi(:,n+1,2) = wi(:,n+1,2)+eta*xic_2*errc_2
            wi(:,n+1,3) = wi(:,n+1,3)+eta*xic_3*errc_3
            wi(:,n+1,4) = wi(:,n+1,4)+eta*xic_4*errc_4
            wj(:,n+1,1) = wj(:,n+1,1)+eta*xir_1*errr_1;   
            wj(:,n+1,2) = wj(:,n+1,2)+eta*xir_2*errr_2;
            wj(:,n+1,3) = wj(:,n+1,3)+eta*xir_3*errr_3;
            wj(:,n+1,4) = wj(:,n+1,4)+eta*xir_4*errr_4;
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
        distr = zeros(4,10);
        distc = zeros(4,10);
        fname = sprintf('digit_%1d_%03d.bmp', CL_fact-1, c+Ntrain);
        input = double(imread(['/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/' fname]))./255;
        A(:,:,1) = input(1:W/sp,1:W/sp);
        A(:,:,2) = input(1:W/sp,W/sp+1:W);
        A(:,:,3) = input(W/sp+1:W,1:W/sp);
        A(:,:,4) = input(W/sp+1:W,W/sp+1:W);
        Ac = reshape(sum(A)./28,4,14);
        distc(1,:) = Ac(1,:)*wi(:,:,1);
        distc(2,:) = Ac(2,:)*wi(:,:,2);
        distc(3,:) = Ac(3,:)*wi(:,:,3);
        distc(4,:) = Ac(4,:)*wi(:,:,4);
        Ar = reshape(sum(A,2)./28,4,14);
        distr(1,:) = Ar(1,:)*wj(:,:,1);
        distr(2,:) = Ar(2,:)*wj(:,:,2);
        distr(3,:) = Ar(3,:)*wj(:,:,3);
        distr(4,:) = Ar(4,:)*wj(:,:,4);        
        dist = sum(distc)+sum(distr);
        [y,indmin] = max(dist);
        confusion(CL_fact,indmin) = confusion(CL_fact,indmin)+1;
    end
end
fprintf('Total Accuracy = %2.1f%%\n',100*sum(diag(confusion))/sum(sum(confusion)));