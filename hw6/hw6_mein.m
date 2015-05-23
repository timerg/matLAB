clear all;
close all;


% fname = /
A = double(imread('Users/TimerPro/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_100.bmp'));
xi = reshape(A,784,1);
hj = zeros(50,1);
