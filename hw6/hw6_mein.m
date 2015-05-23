clear all;
close all;


fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',100);
A = double(imread(fname));
xi = reshape(A,784,1);
hj = zeros(50,1);
