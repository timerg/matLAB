clear; close all;
%% define the parameter
M = 2;
N = 2000;
% U = IDINPUT(N, 'sine', [0 1], [-0.9 0.9]);
U = -1:2/N:1-2/N; U = U(:);
% U = rand(N,1)
bias = 1
one = ones(N,1);
input_all=[one U];
ydes = abs(U);
eta = 0.9

%% input
wji = [1,-1;-2,1]  %teacher's
% wji = [0.490006105630536,0.837346628511838;0.285398982784519,0.180837866239722] %test
% wji = zeros(2,2);  %zero input
% wji = rand(2,2); %randon input
yj = zeros(1,2);
v = zeros(1,2)
wkj = randn(3,1);
%% 
L=5000;
for ii=1:L;
   s = ceil(rand*N);
      x = U(s) ;
      input = [bias x];
      d = ydes(s);    
   v = input*wji;
   yj = 1./(1.+exp(-v));
   
   yk = [bias yj]*wkj;
   err = d-yk;
   wkj_sep = wkj(2:3,:); %two inputa
%    wkj_sep = wkj(2,:); %one input
   delta = err.*wkj_sep.*yj'.*(1.-yj');
   wji = wji + eta*input'*delta';
   wkj = wkj + eta*err*[bias yj]';
   yout(ii) = yk;
   d_all(:,ii) = d;
   x_all(ii) = x;
   wji1_all(ii) = wji(1,1);
   wji2_all(ii) = wji(2,1);
   wji3_all(ii) = wji(1,2);
   wji4_all(ii) = wji(2,2);
% define plot
   w_weight = wji(2,:);
   if mod(ii,10)==0,
        subplot(231); plot(ii,err,'g*');ylabel('error'); hold on;
        subplot(232); plot(ii,wji(2,1),'c.',ii,wji(2,2),'r.');ylabel('weighti');hold on;
        subplot(233); plot(ii,wji(1,1),'c.',ii,wji(1,2),'r.');ylabel('biasi');hold on;
        subplot(234); plot(x,d,'b.',x,yk,'r.'); hold on;
        subplot(235); plot(ii,wkj(1,1),'c.',ii,wkj(2,1),'r.',ii,wkj(3,1),'black.');ylabel('weighti');hold on;
        drawnow;
    end
end
figure(2);
plot(x_all,d_all,'b.',x_all,yout,'r.'); ylabel('ydes'); %hold on;
figure(3);
v_final = input_all*wji;
yj_final = 1./(1.+exp(-v_final));
yk_final = [one yj_final]*wkj;
plot(U, yk_final, 'r.', U,ydes,'b.'); ylabel('final output');
figure(4);
plot(U, yk_final, 'r.'); ylabel('final output');
% set(h,'linewidth',1); hold on;
% set(gca,'ylim',[-1 1.5]);
% plot(x_all,d_all,'.'); ylabel('ydes')
% plot(yout);

    