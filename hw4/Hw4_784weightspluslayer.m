


clear all; close all;

Ntrain = 699; %800 pictures seperate into two parts
Ntest = 99;
W = 28; %28*28 picso

use_importweight = 1;
msg = 'nothing'
if use_importweight==0,
    wj = roundn(rand(W^2,20).*0.2,-3);
    wk = roundn(rand(20,10),-3);
    wk_ini = wk;
    wj_ini = wj;
elseif use_importweight == 1,
    wj_ini = importdata('C:\Users\timer\Documents\GitHub\matlab\hw4\wj_ini.mat')
    wk_ini = importdata('C:\Users\timer\Documents\GitHub\matlab\hw4\wk_ini.mat')
    wj = roundn(wj_ini,-3);
    wk = roundn(wk_ini,-3);
elseif use_importweight == 2,
    wj = roundn(wj_ini.*0.1,-3);
    wk = wk_ini;
else
    error(msg)
end
Amean = ones(W^2,10);
etak = 0.01;
etaj = 0.5;
%% read images and calculate the mean image for each digit
for ll = 0:Ntrain*50
    % for k = 0;
        % for n = 1:1;
        ll = ll
        n = ceil(rand*10)-1
        c = ceil(rand*0.7*1000)-1
            d = zeros(1,10);
            d(n+1) = 1;

% for c = [0:(k*100-1),(k*100+100):(Ntrain+100)];
                fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',n,c);
                B = double(imread(fname));
                xj = reshape(B./255,784,1);

                yj = xj'*wj./784;%wj':1x20
                xk = 1./(1.+exp(-yj));  %1x20
                yk = xk*wk./25;% 1x10;
                err = d-yk;
                % errj = (d-yk)+0.1*sum(sum(wj.^2));

%%% change all

                for nn = 0:9;
                    delta = err(nn+1)*wk(:,nn+1).*xk'.*(1.-xk');
                    wj = wj + etaj.*xj*delta';
                    wj = wj + etaj.*xj*delta'.*(ones(784,20)+(0.0001*sum(sum(wj.^2)))/15680);
                    wk(:,nn+1) = wk(:,nn+1) + etak*err(:,nn+1).*xk';

                end
                % if max(abs(wj))>5,
                %     wj = wj.*0.1;
                % else
                %     wj = wj;
                % end
            % end
        % end
    % end
    % if mod(ll,10)==0,
    %     for sub = 1:10
    %         figure(2); subplot(4,3,sub); plot(ll,err(:,sub),'g.');hold on;drawnow;
    %     end
    % end
end
        for n = 0:9;
            ww = wj*wk;
            Amean(:,n+1) = reshape(ww(:,n+1).*255/10,W^2,1);
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
figure(2);
mesh(linspace(1,20,20),linspace(1,784,784),wj);
