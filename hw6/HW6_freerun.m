% Restricted Boltzmann Machine free run experiments
% YWL
% May 13, 2015

clear; close all; clc;
DIR = '~/Desktop/NN/';
fname = 'scarlett-sq.png';
v0 = imread([DIR fname]);
v0 = imresize(v0,[28 28]);
v0 = double(rgb2gray(v0))/256;
T = 400; % initial temperature relative to training. Empirically, 
         % T=400 works well for "Digit2", and T=60 works well for "all digits". 
th = 0.65; % binary threshold to convert from gray-level to binary

figure(1);
subplot(241); imshow(v0); xlabel('original');
v0 = double(v0<th);
v = v0/T;

figure(1);
subplot(242); imshow(v0); xlabel('v0');

%load('weight500nodesAllDigits.mat');
load('weight20nodesDigit2.mat');

numNodes = size(W,3);
numIter = 10000;
DeltaE = zeros(numNodes,1);
DeltaE_h2v = zeros(28,28);
sw.meanfield = 0; % mean field approximation?


for c = 1:numIter
    % v to h
    for k = 1:numNodes
        DeltaE(k) = sum(sum(W(:,:,k).*v));
    end
    p = 1./(1+exp(-DeltaE)); % probability(h==1)
    tmp = rand(numNodes,1);
    h = (tmp < p);
    if sw.meanfield,
        h = p;
    end
    fprintf('num hidden node on = %d\n',sum(h));
    if sum(h) == 0
        break
    end
    % h to v
    for ii = 1:28
        for jj = 1:28
            w_h = squeeze(W(ii,jj,:));
            DeltaE_h2v(ii,jj) = w_h'*h;
            % energy decrement if v(i,j) drops from 0 to 1
        end
    end
    q = 1./(1+exp(-DeltaE_h2v)); % probability(v(i,j)==1)
    v = (rand(28,28) < q);
    if sum(sum(v))==0,
        break
    end
    if c <= 4,
        figure(1); subplot(2,4,c+4); imshow(imresize(v,[140 140]));
        xlabel(sprintf('v%1d',c));
    else
    
    figure(2);
        imshow(imresize(v,[140 140])); xlabel('v^{(n)}')
        %pause;
    end
end
