clear all
% wire_normal = ['L2-1','L2-3','L2-4','L2-6','L2-7','L2-8','L2-12'];
% wire_abnormal = ['L2-2';'L2-9';'L2-10'];
Vgs = [1.40E+00,1.60E+00,1.80E+00,2.00E+00,2.20E+00,2.40E+00,2.60E+00,2.80E+00,3.00E+00];
mean = 1
devi = 2
V = size(Vgs,2);
    onepr = [-4.14E-12	3.99E-11	7.03E-10	1.04E-08	7.08E-08	2.42E-07	5.59E-07	1.02E-06	1.56E-06;
            5.95E-13	7.85E-12	1.57E-10	1.40E-09	6.80E-09	1.25E-08	2.15E-08	3.04E-08	2.65E-08];
    tenpr = [-8.33E-12	-1.75E-12	6.50E-11	8.74E-10	1.06E-08	6.33E-08	2.15E-07	5.03E-07	9.30E-07;
            3.93E-13	1.24E-12	1.70E-11	2.38E-10	2.12E-09	7.67E-09	1.69E-08	2.68E-08	4.48E-08];
    onenr = [-9.21E-12	-4.13E-12	5.31E-11	7.17E-10	8.86E-09	5.69E-08	2.03E-07	4.64E-07	7.87E-07;
             5.05E-13	8.39E-14	6.25E-12	2.91E-11	4.20E-10	1.86E-09	5.03E-09	1.16E-08	1.97E-08];
figure(1);
plot(Vgs,onepr(1,:),'r-',Vgs,tenpr(1,:),'g-',Vgs,onenr(1,:),'b-');title('r:1pr,g10pr,b1nr');

%%  plot
%one pico
tolerance_1prdown = onepr(mean,:)-onepr(devi,:)-(tenpr(mean,:)+tenpr(devi,:));
tolerance_1prdown_per = tolerance_1prdown./onepr(mean,:);
%ten pico
tolerance_10prup = tolerance_1prdown;
tolerance_10prup_per = tolerance_1prdown_per;
tolerance_10prdown = tenpr(mean,:)-tenpr(devi,:)-(onenr(mean,:)+onenr(devi,:));
tolerance_10prdown_per = tolerance_10prdown./tenpr(mean,:)
%one nano
tolerance_1nrpup = tolerance_10prdown ;
tolerance_1nrup_per = tolerance_10prdown_per ;
%% test
a = zeros(1,9)
b = zeros(1,9)
for iii=1:V
     if  onepr(mean,iii)-tenpr(mean,iii)>0;
        a(iii)=1
     else
        a(iii)=0
    end
     if  tenpr(mean,iii)-onenr(mean,iii)>0;
         b(iii)=1
     else
         b(iii)=0
     end

end

X = Vgs;




% noisemin = min (Z')
% absnoisemin = min((abs(Z')))
figure(2);
subplot(121);plot(X,tolerance_1prdown);title('1pR,10pR_u');
subplot(122);plot(X,tolerance_10prdown);title('10pR_d,1nR');
figure(3);
subplot(121);plot(X,tolerance_1prdown_per);title('1pR,10pR_u');
subplot(122);plot(X,tolerance_10prdown_per);title('10pR_d,1nR');
