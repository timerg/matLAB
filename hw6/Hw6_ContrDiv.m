% Restrictive Boltzmann Machines and the Contrastive Divergence algorithm
% May 10, 2015
% YWLiu
clear; close all;
DIR = '~/GitHub/matlab/hw6';

Ntrain = 2000;
numImages = 1000; % number of examples for each digit
WID = 28; % width of images
HGT = 28; % height of images
sw.learnAllDigits = 0;
sw.meanfield = 0; % this part is under construction, didn't work

if ~sw.learnAllDigits,
    SymbolNum = 2; % which digit to look at
end
numHid = 20; % number of hidden nodes

sigma = 0.4;
W = sigma*randn(WID, HGT, numHid); % weight matrices
%W = zeros(WID,HGT,numHid);

h0 = zeros(numHid,1);
v1 = zeros(WID, HGT);
rho = 0.4;
DeltaE = zeros(numHid,1);
DeltaE_h2v = zeros(WID,HGT);
numRepeats = 1; % number of repeats to estimate statistics

%% read images one by one and conduct contrastive divergence learning
for c = 1:Ntrain
    seednum = floor(numImages*rand);
    if ~sw.learnAllDigits,
        % fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_%1d_%03d.bmp',SymbolNum,seednum);
        fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_%1d_%03d.bmp',SymbolNum,seednum);
    else
        % fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_%1d_%03d.bmp',floor(rand*10),seednum);
        fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_%1d_%03d.bmp',floor(rand*10),seednum);
    end
    v0 = double(imread([fname])>0);
    CorHV0 = zeros(WID, HGT, numHid);

    %% v0->h0
    for k = 1:numHid
        DeltaE(k) = sum(sum(W(:,:,k).*v0));
                    % energy decrement if h(k) drops from 0 to 1
    end
    p = 1./(1+exp(-DeltaE)); % probability(h==1)
    tmp = rand(numHid,1);          %                             Why use random tmp

    CorHV1 = zeros(WID, HGT, numHid); % re-initialize <v1,h1> = 0

    for rr = 1:numRepeats
        h0 = (tmp < p); % simulation repeats several times

        %% h0->v1
        for ii = 1:WID
            for jj = 1:HGT
                w_h = squeeze(W(ii,jj,:));    %the 50 weights connect to the ii,jj_th visible unit
                DeltaE_h2v(ii,jj) = w_h'*h0;
                % energy decrement if v(i,j) drops from 0 to 1
            end
        end
        q = 1./(1+exp(-DeltaE_h2v)); % probability(v(i,j)==1)
        v1 = (rand(WID,HGT) < q);
        %% v1->h1                          WHY  don't put this step into next round
        for k = 1:numHid
            DeltaE(k) = sum(sum(W(:,:,k).*v1));
            % energy decrement if h(k) drops from 0 to 1
        end
        p = 1./(1+exp(-DeltaE)); % probability(h==1)
        tmp = rand(numHid,1);
        h1 = (tmp < p);
        for k = 1:numHid
            if sw.meanfield,
                CovHV1(:,:,k) = CorHV1(:,:,k) + v1*p(k); % mean field?
            else
                CorHV1(:,:,k) = CorHV1(:,:,k) + v1*double(h1(k));
            end
        end
    end
    CorHV1 = CorHV1/numRepeats; % measured correlation <v1,h1>.
    for k = 1:numHid
        CorHV0(:,:,k) = v0*p(k); % hand-calculated expected value of <v0,h0(k)>
    end
    %% Updating the weights
    W = W + rho * (CorHV0 - CorHV1);

    %% monitoring the weight evolution
    figure(1);
    if mod(c,20) == 0,
        % sorting the hidden nodes according to its overall strength of connection
        strength = [zeros(numHid,1) (1:numHid)'];
        for n = 1:numHid
            strength(n,1) = max(max(abs(W(:,:,n))));
        end
        strength = sortrows(strength,1);
        D = abs(strength(end,1));
        figure(1);
        %set(gcf,'position',[40,70,780,570]);
        for n = 1:min(numHid,25)
            subplot(5,5,n);
            imshow(W(:,:,strength(numHid-n+1,2)),[-D D]);
            h=text(2,-2,sprintf('%03d',strength(numHid-n+1,2)));
            set(h,'color','r');
        end
        fprintf('processing image #%1d\n',c);
    end
    if mod(c,100) == 0,
        figure(2)
        subplot(221)
        imshow(v0); xlabel('input')
        subplot(222)
        imshow(v1); xlabel('recons')
        subplot(223)
        imshow(q); xlabel('reconstruction probability')
    end
end
