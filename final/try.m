
% simple
etas = 0.01;
sw = rand(nt, nd*3);
for c = 1:Nbmp
  c
    cc = ceil(rand*999);
    if only2 == 1,
        digit = 2;
        % fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',cc-1);
        fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw6/2_train/digit_2_%03d.bmp',cc);   %for windows
    elseif only2 == 0,
        digit = floor(rand*10);
        % fname = sprintf('~/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',digit,cc-1);
        fname = sprintf('/Users/timer/OneDrive/ms1_2/neuralnetwork/hw4/data/digit_%1d_%03d.bmp',floor(rand*10),cc);
    end
    A = double(imread(fname));
    vi = reshape(A./255, 784, 1);
    vi_origin = vi;

    for aa = 1: 1
        Ejf = (vi_origin)' * wij_w;
        pijf = 1 ./ (1 + exp(-Ejf));
        hj = gt(pijf, rand(1, nh1))';
        Ekf = (hj)' * wjk_w;
        pjkf = 1 ./ (1 + exp(-Ekf));
        hk = gt(pjkf, rand(1, nh2))';   % 500x1
        Etf  = (hk)' * wkt;       %1x2000
        pktf = 1 ./ (1 + exp(-Etf));
        ht = gt(pktf, rand(1, nt))';
        for ss = 1:1
            sh = (ht)' * sw ./ nt;
            sdi = zeros(1,nd*3);
            sdi(digit * 3 +1: digit * 3 +1 +2) = 10;
            ser = sdi - sh;
            sdelta = ht * ser;
            sw = sw + etas .* sdelta;
        end
    end
end

  [sy sp] = max((htt)' * sw ./ nt);
  testresult_s = testresult_s + max([((digit_t + 1)== sp) ((digit_t + 2)== sp) ((digit_t + 3)== sp)]);
  accuracy_s = testresult_s/t_times * 100;

fprintf('accuracy_s = %2.1f%%\n', accuracy_s)
