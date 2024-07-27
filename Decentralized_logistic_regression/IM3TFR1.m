% Matlab Model by Jianghua Yin (Jan.,2022, Nanning)
% Copyright (C) 2022 Jian Group
% All Rights Reserved
%
%% the inertial derivative-free projection method (IDFPM) for solving 
%% constrained nonlinear pseudo-monotone equations of the form
%   F(x)=0, x\in C, 
% where C is a nonempty closed convex set.
%
function [Tcpu,NF,Itr,NormF] = IM3TFR1(fun,method,Switch,model,x0,x0_old,para) 
 
format long

% start the clock
tic;

%% the number of itrations
% Itr=0;   

%% the stopping criterion
epsilon = 1e-6;
epsilon1 = 1e-7;

%% the line search parameters and relaxation factor
k_max = para.Itr_max;   % the maximum number of iterations
gamma = para.gamma;     % the initial guess
sigma = para.sigma;     % the coefficient of line search 
tau = para.tau;         % the compression ratio
% alpha = para.alpha;     % the coefficient of the inertial step
rho = para.rho;         % the relaxation factor 
alpha_try = 0.29;

fprintf('%s & LSmodel=%d & gamma=%.4f & sigma=%.4f & tau=%.4f & Switch=%.4f & rho=%.4f\n', ... 
    method,model,gamma,sigma,tau,Switch,rho);

%% compute the search direction
Fx0 = feval(fun, x0);   % evaluate the function value specified by nprob at x0
NF = 1;  
NormFx0 = norm(Fx0);                   
% x0_old = x0;
L1 = 0;
     
for k=1:k_max
    
    if k==1 && NormFx0<=epsilon
        L1 = 1;
        NormF = NormFx0; % the final norm of equations
        break; 
    end
    if k==1
        alpha = alpha_try;
    else
        if Switch==1
            alpha = min(alpha_try,1/(norm(x0-x0_old)*k^2));
        elseif Switch==2
            alpha = min(alpha_try,1/(k*norm(x0-x0_old))^2);
        else
            alpha = alpha_try;
        end
    end
    %% compute the inertial step %%
    y0 = x0+alpha*(x0-x0_old);
    Fy0 = feval(fun, y0);
    NF = NF+1;
    NormFy0 = norm(Fy0);
    if NormFy0<=epsilon
        L1 = 1;
        NormF = NormFy0;   % the final norm of equations
        break; 
    end
    
    %% compute the initial direction %%
    if k==1
        dk = -Fy0;
    else
        % update the search direction
        switch method
            case 'IGPM'
                dk = -Fy0; 
            case 'IHTTCGPM'
                w0 = Fy0-Fy0_old;
                NormFy0_old = norm(Fy0_old);
                Fy0tw0 = Fy0'*w0;
                Fy0tdk = Fy0'*dk;
                denominator = max(1.4*Normdk*norm(w0),NormFy0_old^2);
                betak = Fy0tw0/denominator;
                thetak = Fy0tdk/denominator;
                dk = -Fy0+betak*dk-thetak*w0;
            case 'IHCGPM'
                w0 = Fy0-Fy0_old;
                NormFy0_old = norm(Fy0_old);
                dktw0 = dk'*w0;
                Fy0tFy0_old = Fy0'*Fy0_old;
                numerator = NormFy0^2-max(0,NormFy0/NormFy0_old*Fy0tFy0_old);
                denominator = max(NormFy0^2,max(3*Normdk*NormFy0,dktw0));   
                betak = numerator/denominator;
                dk = -Fy0+betak*dk;
             case 'MITTCGP'
                w0 = Fy0-Fy0_old;
                sk = y0-y0_old;
                tk = 1+max(0,-dk'*w0/norm(dk)^2);
                wk = w0+tk*dk;
                Tk = min(0.4,max(0,1-w0'*sk/norm(w0)^2)); % yk1=yk+0.01*dk的数值效果没有直接使用yk效果好  
                fenmu = max(norm(Fy0_old)^2,max(0.1*norm(dk)*norm(wk),dk'*wk));
                thetak = Tk*Fy0'*dk/fenmu;  % 原始的分母是 norm(Fk0)^2  
                betak = Fy0'*wk/fenmu-norm(wk)^2*Fy0'*dk/fenmu^2; %(max(mu*norm(dk)*norm(Fk),norm(Fk0)^2))
                dk = -Fy0+betak*dk+thetak*wk;  
            case 'FITTCGPM-PRP'
                w0= Fy0-Fy0_old;
%               betak=(norm(Fk)^2-max(0,norm(Fk)/norm(Fk0)*Fk'*Fk0))/max(norm(Fk0)^2,dk'*yk);
                betak=Fy0'*w0/norm(Fy0_old)^2;
                vk=NormFy0/max(norm(Fy0_old),abs(betak)*norm(dk));
                thetak=-0.055*Fy0'*Fy0_old/norm(Fy0_old)^2; 
                dk=-Fy0+0.001*vk*betak*dk+thetak*Fy0_old;
             case'FITTCGPM-DY'%第二篇
                 w0 = Fy0-Fy0_old;
                betak=(norm(Fy0))^2/(dk'*w0);
                nuk=norm(Fy0)/max(norm(Fy0_old),(abs(betak))*norm(dk));
                thetak=-0.055*Fy0'*Fy0_old/norm(Fy0_old)^2;  % 原始的分母是 norm(Fk0)^2  
                dk=-Fy0+0.001*nuk*betak*dk+thetak*Fy0_old;  
            case 'FITTCGPM-FR'%第二篇
%                 w0 = Fy0-Fy0_old;
                betak=norm(Fy0)^2/norm(Fy0_old)^2;
                nuk=norm(Fy0)/max(norm(Fy0_old),(abs(betak))*norm(dk));
                thetak=-0.055*Fy0'*Fy0_old/norm(Fy0_old)^2;  % 原始的分母是 norm(Fk0)^2  
                dk=-Fy0+0.001*nuk*betak*dk+thetak*Fy0_old;
            case'ISTCP'%Ibrahim, A.H., Kumam, P ., Sun, M., Chaipunya, P .: Projection method with inertial step for nonlinear equations: Application to signal recovery. J. Ind. Manag. Optim. 19(1), 30–55 (2023)
                betak=Fy0'*Fy0_old/(Fy0_old'*dk);
                thetak=Fy0'*dk/(Fy0_old'*dk); 
                dk=-Fy0+betak*dk+thetak*Fy0_old;
            case'IM3TFR1'  
                mk=z0_new-y0;%t*dk;%
                betak=norm(Fy0)^2/norm(Fy0_old)^2;
                thetak=Fy0'*mk/norm(Fy0_old)^2;  % 原始的分母是 norm(Fk0)^2  
                dk=-Fy0+betak*mk-thetak*Fy0;
            otherwise
                disp('Input error! Please check the input method');
        end
    end
    Normdk = norm(dk);
    if Normdk<epsilon1
        L1 = 1;
        NormF = NormFy0;
        break;
    end
    Normdk2 = Normdk^2;
    Fy0_old = Fy0;
    
    %%% Start Armijo-type line search  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % model=1 means -F(zk)'*dk ?? sigma*tk*norm(dk)^2
    % model=2 means -F(zk)'*dk ?? sigma*tk*norm(F(zk))*norm(dk)^2
    % model=3 means -F(zk)'*dk ?? sigma*tk*norm(F(zk))/(1+norm(F(zk)))*norm(dk)^2
    % model=4 means -F(zk)'*dk ?? sigma*tk*max(lambda,min(nu,norm(Fz_new,2)))*norm(dk)^2
    if model==1
        t = gamma;
        z0_new = y0+t*dk;
        Fz0_new = feval(fun, z0_new);
        NF = NF+1;
        Fz0_newtdk = -Fz0_new'*dk;
        % check the Armijo-type line search condition
        while Fz0_newtdk < sigma*t*Normdk2 && t>10^-6  
            % the Armijo-type line search condition violated
            t = t*tau;
            z0_new = y0+t*dk;
            Fz0_new = feval(fun, z0_new);
            NF = NF+1;
            Fz0_newtdk = -Fz0_new'*dk;
        end %%% End Armijo-type line search %%%
        NormFz0_new = norm(Fz0_new);
    elseif model==2
        t = gamma;
        z0_new = y0+t*dk;
        Fz0_new = feval(fun, z0_new);
        NF = NF+1;
        NormFz0_new = norm(Fz0_new);
        Fz0_newtdk = -Fz0_new'*dk;
        % check the Armijo-type line search condition
        while Fz0_newtdk < sigma*t*NormFz0_new*Normdk2 && t>10^-6  
            % the Armijo-type line search condition violated
            t = t*tau;
            z0_new = y0+t*dk;
            Fz0_new = feval(fun, z0_new);
            NF = NF+1;
            NormFz0_new = norm(Fz0_new);
            Fz0_newtdk = -Fz0_new'*dk;
        end %%% End Armijo-type line search %%%
    elseif model==3
        t = gamma;
        z0_new = y0+t*dk;
        Fz0_new = feval(fun, z0_new);
        NF = NF+1;
        NormFz0_new = norm(Fz0_new);
        Fz0_newtdk = -Fz0_new'*dk;
        % check the Armijo-type line search condition
        while Fz0_newtdk < sigma*t*NormFz0_new/(1+NormFz0_new)*Normdk2 && t>10^-6  
            % the Armijo-type line search condition violated
            t = t*tau;
            z0_new = y0+t*dk;
            Fz0_new = feval(fun, z0_new);
            NF = NF+1;
            NormFz0_new = norm(Fz0_new);
            Fz0_newtdk = -Fz0_new'*dk;
        end %%% End Armijo-type line search %%%
    else
        t = gamma;
        z0_new = y0+t*dk;
        Fz0_new = feval(fun, z0_new);
        NF = NF+1;
        NormFz0_new = norm(Fz0_new);
        Fz0_newtdk = -Fz0_new'*dk;
        % check the Armijo-type line search condition
        while Fz0_newtdk < sigma*t*max(0.001,min(0.8,NormFz0_new))*Normdk2 && t>10^-6  
            % the Armijo-type line search condition violated
            t = t*tau;
            z0_new = y0+t*dk;
            Fz0_new = feval(fun, z0_new);
            NF = NF+1;
            NormFz0_new = norm(Fz0_new);
            Fz0_newtdk = -Fz0_new'*dk;
        end %%% End Armijo-type line search %%%
    end 
    Fz0 = Fz0_new;
    NormFz0 = NormFz0_new;
%     if NormFz0<=epsilon
%         L1 = 1;
%         NormF = NormFz0; % the final norm of equations
%         break;
%     end
    xik = t*Fz0_newtdk/NormFz0^2;
    % compute the next iteration 
    x1 = y0-rho*xik*Fz0;
    Fx1 = feval(fun, x1);
    NF = NF+1;
    NormFx1 = norm(Fx1);
    if NormFx1<=epsilon
        L1 = 1;
        NormF = NormFx1;
        break;
    end
    
    % update the iteration
    x0_old = x0;
    y0_old = y0;
    x0 = x1;
    NormFx0 = NormFx1;
end
if L1==1
    Itr = k;
    Tcpu = toc;
else
    NF = NaN;
    Itr = NaN;
    Tcpu = NaN;
    NormF = NaN;
end