clear all
% wire_normal = ['L2-1','L2-3','L2-4','L2-6','L2-7','L2-8','L2-12'];
% wire_abnormal = ['L2-2';'L2-9';'L2-10'];
Vgs = [1.40E+00,1.60E+00,1.80E+00,2.00E+00,2.20E+00,2.40E+00,2.60E+00,2.80E+00,3.00E+00];
V = size(Vgs,2);
    onefr = [2.43E-12	3.52E-12	2.91E-12	3.08E-13	3.18E-12	4.34E-12	4.74E-12;
            1.08E-12	7.50E-13	2.73E-13	1.53E-13	3.73E-13	9.13E-13	9.30E-13;
            4.86E-11	4.73E-11	3.55E-11	8.78E-12	2.49E-11	3.52E-11	1.19E-10;
            1.10E-11	1.16E-11	6.15E-12	1.40E-12	2.90E-12	7.10E-12	9.54E-12;
            6.63E-10	5.95E-10	4.92E-10	8.28E-11	2.26E-10	3.34E-10	2.23E-09;
            1.39E-10	1.60E-10	7.59E-11	1.61E-11	4.02E-11	7.47E-11	2.15E-10;
            8.48E-09	7.33E-09	7.68E-09	1.03E-09	2.81E-09	3.96E-09	1.86E-08;
            1.37E-09	1.69E-09	1.03E-09	2.37E-10	5.87E-10	8.26E-10	1.25E-09;
            5.09E-08	4.89E-08	5.36E-08	1.08E-08	2.48E-08	3.13E-08	7.17E-08;
            5.63E-09	7.85E-09	3.94E-09	1.86E-09	3.91E-09	5.79E-09	5.88E-09;
            1.74E-07	1.75E-07	1.89E-07	5.84E-08	1.13E-07	1.33E-07	1.70E-07;
            1.36E-08	2.35E-08	8.19E-09	6.50E-09	1.23E-08	1.72E-08	6.93E-09;
            4.11E-07	4.32E-07	4.17E-07	1.88E-07	3.25E-07	3.62E-07	2.89E-07;
            2.50E-08	4.12E-08	9.64E-09	1.08E-08	2.11E-08	2.76E-08	1.26E-08;
            7.51E-07	8.22E-07	6.74E-07	4.33E-07	6.68E-07	7.37E-07	4.00E-07;
            3.15E-08	6.22E-08	1.17E-08	2.10E-08	2.66E-08	4.20E-08	2.05E-08;
            1.17E-06	1.31E-06	9.03E-07	7.85E-07	1.11E-06	1.21E-06	5.01E-07;
            3.21E-08	7.02E-08	1.75E-08	2.55E-08	3.51E-08	3.51E-08	8.14E-09];
hundredfr =[4.85E-12	7.18E-12	8.76E-12	5.27E-13	2.01E-12	2.29E-12	1.24E-12;
            1.15E-12	1.90E-12	2.35E-12	1.75E-13	3.82E-13	5.94E-13	7.75E-13;
            8.83E-11	8.31E-11	9.95E-11	6.29E-12	1.52E-11	1.92E-11	7.31E-11;
            1.47E-11	1.84E-11	2.93E-11	1.47E-12	3.01E-12	4.27E-12	1.27E-11;
            1.32E-09	1.18E-09	1.59E-09	4.77E-11	1.37E-10	1.61E-10	1.50E-09;
            2.62E-10	2.83E-10	4.97E-10	1.09E-11	2.65E-11	3.74E-11	3.02E-10;
            1.51E-08	1.36E-08	1.82E-08	4.43E-10	1.75E-09	1.84E-09	1.43E-08;
            2.37E-09	2.64E-09	4.29E-09	1.21E-10	5.11E-10	4.87E-10	1.78E-09;
            7.67E-08	7.31E-08	9.16E-08	4.58E-09	1.66E-08	1.65E-08	6.14E-08;
            8.20E-09	9.56E-09	1.36E-08	1.20E-09	3.56E-09	3.11E-09	4.35E-09;
            2.30E-07	2.34E-07	2.75E-07	3.12E-08	8.50E-08	8.15E-08	1.55E-07;
            1.66E-08	2.35E-08	2.79E-08	5.24E-09	1.06E-08	1.23E-08	7.55E-09;
            5.00E-07	5.38E-07	5.97E-07	1.24E-07	2.63E-07	2.52E-07	2.66E-07;
            2.62E-08	3.40E-08	4.03E-08	1.67E-08	2.15E-08	2.55E-08	1.15E-08;
            8.76E-07	9.70E-07	1.04E-06	3.18E-07	5.73E-07	5.58E-07	3.71E-07;
            3.25E-08	4.81E-08	5.39E-08	3.25E-08	2.72E-08	3.75E-08	5.77E-09;
            1.31E-06	1.48E-06	1.55E-06	5.83E-07	1.01E-06	9.66E-07	4.61E-07;
            3.11E-08	5.00E-08	5.03E-08	3.97E-08	3.61E-08	4.45E-08	1.12E-08]  ;
L = size(onefr,2);
for iii=1:V
    v_2 = 2*iii;
    for ii=1:L
     if  onefr(v_2-1,ii)-hundredfr(v_2-1,ii)>0;
        tolerance = onefr(v_2-1,ii)-onefr(v_2,ii)-(hundredfr(v_2-1,ii)+hundredfr(v_2,ii));
        tolerance_percentage = (onefr(v_2-1,ii)-onefr(v_2,ii)-(hundredfr(v_2-1,ii)+hundredfr(v_2,ii)))/onefr(v_2-1,ii);
        a=1
     else
         tolerance =-( hundredfr(v_2-1,ii)-hundredfr(v_2,ii)-(onefr(v_2-1,ii)+onefr(v_2,ii)));
         tolerance_percentage = -(hundredfr(v_2-1,ii)-hundredfr(v_2,ii)-(onefr(v_2-1,ii)+onefr(v_2,ii)))/hundredfr(v_2-1,ii);
        a=0
     end
        tolerance_vgs(ii) = tolerance;
        tolerance_percentagevgs(ii) = tolerance_percentage;;
        a_l(ii)=a;
   
    
    end
   tolernace_all(iii,:) = tolerance_vgs;
   tolerance_percentageall(iii,:) = tolerance_percentagevgs;
   
   a_all(iii,:) = a_l
   
end
Y = Vgs;
X = linspace(1,L,L);
Z = tolernace_all;
Z2 = tolerance_percentageall

noisemin = min (Z')
absnoisemin = min((abs(Z')))


figure(1);
mesh(X,Y,Z); xlabel('L2-X');ylabel('Vgs');zlabel('A');
set(gca,'zlim',[0 0.5E-7]);
figure(2);
mesh(X,Y,Z2); xlabel('L2-X');ylabel('Vgs');zlabel('%');
% set(gca,'zlim',[0 0.5E-7])
figure(3);
subplot(231);plot(X,Z(8,:));xlabel('vgs=2.8');
subplot(232);plot(X,Z(9,:));xlabel('vgs=3');
subplot(233);plot(X,Z(7,:));xlabel('vgs=2.6');
subplot(234);plot(X,Z(6,:));xlabel('vgs=2.4');
subplot(235);plot(X,Z(5,:));xlabel('vgs=2.2');
figure(4);
subplot(247);plot(Y,Z2(:,7));xlabel('L2-12');
subplot(246);plot(Y,Z2(:,6));xlabel('L2-8');
subplot(245);plot(Y,Z2(:,5));xlabel('L2-7');
subplot(244);plot(Y,Z2(:,4));xlabel('L2-6');
subplot(243);plot(Y,Z2(:,3));xlabel('vL2-4');
subplot(242);plot(Y,Z2(:,2));xlabel('L2-3');
subplot(241);plot(Y,Z2(:,1));xlabel('L2-1');



