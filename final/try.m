avg = zeros(82,1);
for cc=1:82
fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/final/H%01d.bmp',cc)
xx = double(imread(fname))
avg(cc) = sum(sum(xx))/32/32
