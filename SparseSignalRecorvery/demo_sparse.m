% This is a demo of DFPM for solving \ell_1-norm minimization
% problem
% arg min_x = 0.5*|| A x - b ||_2^2 + lambda || x ||_1
% using the algorithm modified three-term conjugate gradient projection method, described in the following paper
%
% -----------------------------------------------------------------------
% Copyright (2021): Jianghua Yin
% ----------------------------------------------------------------------
%
% The first version of this code by Jianghua Yin, June., 30, 2021

% If possible, please cite the following papers:
% [1] Yin J, Jian J, Jiang X, et al. A hybrid three-term conjugate gradient 
% projection method for constrained nonlinear monotone equations with applications. 
% Numerical Algorithms, 2021, 88(1): 389-418.
% [2] Yin J, Jian J, Jiang X. A generalized hybrid CGPM-based algorithm for solving 
% large-scale convex constrained equations with applications to image restoration. 
% Journal of Computational and Applied Mathematics, 2021, 391: 113423.
% [3] 尹江华, 简金宝, 江羡珍. 凸约束非光滑方程组基于自适应线搜索的谱梯度投影算法. 
% 计算数学, 2020, 42(4): 457-471.


clear all
close all
% clc
%set parameters
% randn('seed',1); % 1 for the experiments in the paper
% rand('seed',1); % 1 for the experiments in the paper
rng(2016);
% rng('default')

% addpath('data') 

ITR_max = 2000;
% parameters for IRSDGPM
para1.Itr_max = ITR_max;
para1.gamma = 1;         % the initial guess
para1.sigma = 0.01;      % the coefficient of line search 
para1.tau = 0.5;         % the compression ratio
para1.alpha = 0.18966;   % the coefficient of inertial step
para1.rho = 1.4882;      % the relaxation factor 
% para1.alpha = 0.1;   % the coefficient of inertial step
% para1.rho = 1.7606;      % the relaxation factor 

% parameters for USDGPM
para2.Itr_max = ITR_max;
para2.gamma = 1;         % the initial guess
para2.sigma = 0.01;      % the coefficient of line search 
para2.tau = 0.5;         % the compression ratio
para2.rho = 1;           % the relaxation factor 

% parameters for MITTCGP
para3.Itr_max = ITR_max;                                 
para3.gamma = 0.4;         % the initial guess
para3.sigma = 0.01;      % the coefficient of line search 
para3.tau = 0.6;         % the compression ratio
para3.alpha = 0.1;       % the coefficient of inertial step
para3.rho = 1.8;      % the relaxation factor 
% para3.alpha = 0.1;       % the coefficient of inertial step
% para3.rho = 1.7606;      % the relaxation factor 

% parameters for UT2CGPM
para4.Itr_max = ITR_max;
para4.gamma = 1;         % the initial guess
para4.sigma = 0.01;      % the coefficient of line search 
para4.tau = 0.5;         % the compression ratio
para4.rho = 1;           % the relaxation factor 

% parameters for ISTCP
para5.Itr_max = ITR_max;
para5.gamma = 1;         % the initial guess
para5.sigma = 0.0001;      % the coefficient of line search 
para5.tau = 0.7;         % the compression ratio
para5.alpha = 0.8;       % the coefficient of inertial step
para5.rho = 1;      % the relaxation factor 
% para5.alpha = 0.1;       % the coefficient of inertial step
% para5.rho = 1.7606;      % the relaxation factor 

% parameters for IM3TFR1
para6.Itr_max = ITR_max;
para6.gamma = 1;         % the initial guess
para6.sigma = 0.01;      % the coefficient of line search 
para6.tau = 0.5;         % the compression ratio
para6.alpha = 0.29;       % the coefficient of inertial step
para6.rho = 1;      % the relaxation factor 
% n is the original signal length
n = 2^12;

% k is number of observations to make
m = 2^10;

% number of spikes to put down
% n_spikes = floor(.01*n);
n_spikes = 64;


% random +/- 1 signal
f = zeros(n,1);
q = randperm(n);
% f(q(1:n_spikes)) = sign(randn(n_spikes,1));
f(q(1:n_spikes)) = randn(n_spikes,1);

% measurement matrix
disp('Creating measurement matrix...');
R = randn(m,n);

% orthonormalize rows
R = orth(R')';

if n == 8192  
   % in this case, we load a precomputed
   % matrix to save some time
   load Rmatrix_2048_8192.mat
end
%
disp('Finished creating matrix');

hR = @(x) R*x;
hRt = @(x) R'*x;

% noisy observations
sigma = 0.001; 
b = hR(f) + sigma*randn(m,1);

% regularization parameter
varrho = 0.005*max(abs(R'*b));

% % initial point
% f0 = hRt(b);

disp('Starting MITTCGP')
[x1,mses1,Tcpu1,NF1,NormF1] = IRDFPM(R,b,varrho,f,'MITTCGP',4,para3);
T1 = Tcpu1(end);
% IRSDGPM_mses = mses1(end)

disp('Starting IRTTCGPM-PRP')
[x2,mses2,Tcpu2,NF2,NormF2] = IRDFPM(R,b,varrho,f,'FITTCGPM-PRP',1,para3);
T2 = Tcpu2(end);
% USDGPM_mses = mses2(end)

disp('Starting IRTTCGPM-DY')
[x3,mses3,Tcpu3,NF3,NormF3] = IRDFPM(R,b,varrho,f,'FITTCGPM-DY',1,para3);
T3 = Tcpu3(end);
% USDGPM_mses = mses2(end)

disp('Starting IM3TFR1')
[x4,mses4,Tcpu4,NF4,NormF4] = IRDFPM(R,b,varrho,f,'IM3TFR1',2,para6);
T4 = Tcpu4(end);

% disp('Starting ISTCP')
% [x4,mses4,Tcpu4,NF4,NormF4] = IRDFPM(R,b,varrho,f,'ISTCP',2,para5);
% T4 = Tcpu4(end);
% IRT2CGPM_mses = mses3(end)

%
fprintf(1,'\n\n-------------------------------------------------\n')   
fprintf(1,'-------------------------------------------------\n')   
fprintf(1,'Problem: n = %g,  m = %g, number of spikes = %g, varrho = %g\n',n,m,n_spikes,varrho)
fprintf(1,'All algorithms initialized with Atb\n')
fprintf(1,'-------------------------------------------------\n')


fprintf(1,'\n MITTCGP Tcpu: %6.2f secs (%d iterations), MSE of the solution = %6.5e\n',...
        T1,length(mses1),mses1(end))  
fprintf(1,'\n FITTCGPM-PRP Tcpu: %6.2f secs (%d iterations), MSE of the solution = %6.5e\n',...
        T2,length(mses2),mses2(end))
fprintf(1,'\n FITTCGPM-DY Tcpu: %6.2f secs (%d iterations), MSE of the solution = %6.5e\n',...
        T3,length(mses3),mses3(end))    
fprintf(1,'\n IM3TFR1 Tcpu: %6.2f secs (%d iterations), MSE of the solution = %6.5e\n',...
        T4,length(mses4),mses4(end))   
% fprintf(1,'\n ISTCP Tcpu: %6.2f secs (%d iterations), MSE of the solution = %6.5e\n',...
%         T4,length(mses4),mses4(end))  
% fprintf(1,'\n UT2CGPM Tcpu: %6.2f secs (%d iterations), MSE of the solution = %6.3e\n',...
%         T4,length(mses4),mses4(end))
   
fprintf(1,'-------------------------------------------------\n')
fprintf(1,'-------------------------------------------------\n')

% ================= Plotting results =================
%%  semilogy
figure(1)
plot(mses1,'b-','LineWidth',2)
hold on
plot(mses2,'r-','LineWidth',2)
hold on
plot(mses3,'g-','LineWidth',2)
hold on
plot(mses4,'k-','LineWidth',2)
hold on
legend('MITTCGP','FITTCGPM-PRP','FITTCGPM-DY','IM3TFR1');%,'TTCGPM') 
set(gca,'FontName','Times','FontSize',16)
xlabel('Itr')
ylabel('MSE')
title(sprintf('m=%d, n=%d, N=%d, varrho=%.2e',m,n,n_spikes,varrho))
axis on
grid on
hold off

figure(2)
plot(Tcpu1,mses1,'b-','LineWidth',2)
hold on
plot(Tcpu2,mses2,'r-','LineWidth',2)
hold on
plot(Tcpu3,mses3,'g-','LineWidth',2)
hold on
plot(Tcpu4,mses4,'k-','LineWidth',2)
hold on
% plot(Tcpu4,mses4,'k-','LineWidth',2)
% hold on
legend('MITTCGP','FITTCGPM-PRP','FITTCGPM-DY','IM3TFR1');%,'TTCGPM')
set(gca,'FontName','Times','FontSize',16)
xlabel('Tcpu')
ylabel('MSE')
title(sprintf('m=%d, n=%d, N=%d, varrho=%.2e',m,n,n_spikes,varrho))
axis on
grid on
hold off
% %==========================================================================
% figure(3)
% plot(obj_SGCS_cont,'b:','LineWidth',2)
% hold on
% plot(obj_CGDCS_cont,'r-','LineWidth',2)
% legend('HTTCGP-CS','CGD')
% set(gca,'FontName','Times','FontSize',16)
% xlabel('Iterations')
% ylabel('ObjFun')
% title(sprintf('n=%d, k=%d, tau=%g',n,k,tau))
% axis on
% grid on
% hold off
% 
% figure(4)
% plot(times_SGCS_cont,obj_SGCS_cont,'b:','LineWidth',2)
% hold on
% plot(times_CGDCS_cont,obj_CGDCS_cont,'r-','LineWidth',2)
% hold on
% legend('HTTCGP-CS','CGD')
% set(gca,'FontName','Times','FontSize',16)
% xlabel('CPU time (seconds)')
% ylabel('ObjFun')
% title(sprintf('n=%d, k=%d, tau=%g',n,k,tau))
% axis on
% grid on
% hold off
%==========================================================================
%%
figure(3)
scrsz = get(0,'ScreenSize');
% set(7,'Position',[10 scrsz(4)*0.1 0.9*scrsz(3)/2 3*scrsz(4)/4])
subplot(5,1,1)
plot(f,'LineWidth',1.1)
top = max(f(:));
bottom = min(f(:));
v = [0 n+1 bottom-0.05*(top-bottom)  top+0.05*((top-bottom))];
set(gca,'FontName','Times')
set(gca,'FontSize',14)
% set(get(gca,'title'),'fontname','宋体')
% title(['原始稀疏信号 n=',num2str(n),' N=',num2str(n_spikes)])
title(sprintf('Original (m = %g, n = %g, N = %g)',m,n,n_spikes))
axis(v)

% % scrsz = get(0,'ScreenSize');
% % set(7,'Position',[10 scrsz(4)*0.1 0.9*scrsz(3)/2 3*scrsz(4)/4])
% subplot(6,1,2)
% plot(b,'LineWidth',1.1)
% top = max(b(:));
% bottom = min(b(:));
% v = [0 m+1 bottom-0.05*(top-bottom)  top+0.05*((top-bottom))];
% set(gca,'FontName','Times')
% set(gca,'FontSize',14)
% title(sprintf('b'))
% axis(v)

scrsz = get(0,'ScreenSize');
% set(7,'Position',[10 scrsz(4)*0.1 0.9*scrsz(3)/2 3*scrsz(4)/4])
subplot(5,1,2)
plot(x1(:),'LineWidth',1.1)
top = max(x1(:));
bottom = min(x1(:));
v = [0 n+1 bottom-0.05*(top-bottom)  top+0.05*((top-bottom))];
set(gca,'FontName','Times')
set(gca,'FontSize',14)
title(sprintf('MITTCGP (MSE = %5.6e, Itr=%g, Tcpu=%4.2fs)',  mses1(end),length(mses1),Tcpu1(end)))
axis(v)

scrsz = get(0,'ScreenSize');
% set(7,'Position',[10 scrsz(4)*0.1 0.9*scrsz(3)/2 3*scrsz(4)/4])
subplot(5,1,3)
plot(x2(:),'LineWidth',1.1)
top = max(x2(:));
bottom = min(x2(:));
v = [0 n+1 bottom-0.05*(top-bottom)  top+0.05*((top-bottom))];
set(gca,'FontName','Times')
set(gca,'FontSize',14)
title(sprintf('FITTCGPM-PRP (MSE = %5.6e, Itr=%g, Tcpu=%4.2fs)',mses2(end),length(mses2),Tcpu2(end)))
axis(v)

scrsz = get(0,'ScreenSize');
% set(7,'Position',[10 scrsz(4)*0.1 0.9*scrsz(3)/2 3*scrsz(4)/4])
subplot(5,1,4)
plot(x3(:),'LineWidth',1.1)
top = max(x3(:));
bottom = min(x3(:));
v = [0 n+1 bottom-0.05*(top-bottom)  top+0.05*((top-bottom))];
set(gca,'FontName','Times')
set(gca,'FontSize',14)
title(sprintf('FITTCGPM-DY (MSE = %5.6e, Itr=%g, Tcpu=%4.2fs)',mses3(end),length(mses3),Tcpu3(end)))
axis(v)

scrsz = get(0,'ScreenSize');
% set(7,'Position',[10 scrsz(4)*0.1 0.9*scrsz(3)/2 3*scrsz(4)/4])
subplot(5,1,5)
plot(x4(:),'LineWidth',1.1)
top = max(x4(:));
bottom = min(x4(:));
v = [0 n+1 bottom-0.05*(top-bottom)  top+0.05*((top-bottom))];
set(gca,'FontName','Times')
set(gca,'FontSize',14)
title(sprintf('IM3TFR1 (MSE = %5.6e, Itr=%g, Tcpu=%4.2fs)',mses4(end),length(mses4),Tcpu4(end)))
axis(v)

% % scrsz = get(0,'ScreenSize');
% % set(7,'Position',[10 scrsz(4)*0.1 0.9*scrsz(3)/2 3*scrsz(4)/4])
% subplot(5,1,5)
% plot(x4(:),'LineWidth',1.1)
% top = max(x4(:));
% bottom = min(x4(:));
% v = [0 n+1 bottom-0.05*(top-bottom)  top+0.05*((top-bottom))];
% set(gca,'FontName','Times')
% set(gca,'FontSize',14)
% title(sprintf('TTCGPM (MSE = %5.2e, Itr=%g, Tcpu=%4.2fs)',mses4(end),length(mses4),Tcpu4(end)))
% axis(v)
% % 
% % 
