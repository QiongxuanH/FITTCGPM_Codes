% Matlab Model by Jianghua Yin (July,2021, Nanning)
% Copyright (C) 2021 Jian Group
% All Rights Reserved
%
%% the inertial-relaxed derivative-free projection method (IRDFPM) for solving the 
%% unconstrained nonlinear monotone equations of the form
%   F(x)=0. 
%
function [x1,mses,Tcpu,NF,NormF] = IRDFPM(A,b,lambda,true_x,method,model,para) 
 
format long

% if A is a matrix, we find out dimensions of y and x,
% and create function handles for multiplication by A and A',
% so that the code below doesn't have to distinguish between
% the handle/not-handle cases
if ~isa(A, 'function_handle')
  AT = @(x) (x'*A)'; %A'*x;
  A = @(x) A*x;
end
% from this point down, A and AT are always function handles.

% Precompute A'*b since it'll be used a lot
Atb = AT(b);    
x0 = Atb;
n = length(x0);

% initial point
xu0 =  x0.*(x0 >= 0);
xv0 = -x0.*(x0 <  0);
% from these two relations, x0 = u0-v0;

%% the stopping criterion
epsilon = 1e-6;
epsilon1 = 1e-7;

%% the line search parameters and relaxation factor
k_max = para.Itr_max;   % the maximum number of iterations
gamma = para.gamma;     % the initial guess
sigma = para.sigma;     % the coefficient of line search 
tau = para.tau;         % the compression ratio
alpha = para.alpha;     % the coefficient of the inertial step
rho = para.rho;         % the relaxation factor 

fprintf('%s & LSmodel=%d & gamma=%.4f & sigma=%.4f & tau=%.4f & alpha=%.4f & rho=%.4f\n', ... 
    method,model,gamma,sigma,tau,alpha,rho);

% start the clock
t0 = cputime;

% % define function handle
% Fu = @(x,xu) min(xu,AT(A(x))-Atb+lambda);
% Fv = @(x,xv) min(xv,-AT(A(x))+Atb+lambda);

%% compute the search direction
Ax0 = A(x0);
tempx = AT(Ax0)-Atb;
Fxu0 = min(xu0,tempx+lambda);
Fxv0 = min(xv0,-tempx+lambda);
NF = 1;  
NormFxk2 = Fxu0'*Fxu0+Fxv0'*Fxv0;    
NormFxk = sqrt(NormFxk2);
xu0_old = xu0;
xv0_old = xv0;
L1 = 0;
     
for k=1:k_max
    
    if k==1 && NormFxk<=epsilon
        L1 = 1;
        NormF = NormFxk; % the final norm of equations
        mses(k) = 1/n*norm(x0-true_x)^2;
        Tcpu(k) = cputime-t0;
        break; 
    end
    
    %% compute the inertial step %%
    yuk = xu0+alpha*(xu0-xu0_old);
    yvk = xv0+alpha*(xv0-xv0_old);
    yk = yuk-yvk;
    Ayk = A(yk);
    tempy = AT(Ayk)-Atb;
    Fyuk = min(yuk,tempy+lambda);
    Fyvk = min(yvk,-tempy+lambda);
    NF = NF+1;
    NormFyk2 = Fyuk'*Fyuk+Fyvk'*Fyvk;
    NormFyk = sqrt(NormFyk2);
    if NormFyk<=epsilon
        L1 = 1;
        NormF = NormFyk;   % the final norm of equations
        mses(k) = 1/n*norm(yk-true_x)^2;
        Tcpu(k) = cputime-t0;
        break; 
    end
    
    %% compute the initial direction %%
    if k==1
        dku = -Fyuk;
        dkv = -Fyvk;
    else
        % update the search direction
        switch method
            case 'IRGPM'
                dku = -Fyuk;
                dkv = -Fyvk; 
            case 'IRTTCGPM'
                wuk = Fyuk-Fyuk_old;
                wvk = Fyvk-Fyvk_old;
                Fyktwk = Fyuk'*wuk+Fyvk'*wvk;
                Fyktdk = Fyuk'*dku+Fyvk'*dkv;
                Normwk2 = wuk'*wuk+wvk'*wvk;
                Normwk = sqrt(Normwk2);
                Den = max(1.4*Normdk*Normwk,NormFyk2_old);
                betak = Fyktwk/Den;
                lambdak = Fyktdk/Den;
                dku = -Fyuk+betak*dku-lambdak*wuk;
                dkv = -Fyvk+betak*dkv-lambdak*wvk;
            case'IRTTCGPMN'
                 suk = yuk-yuk_old;
                 svk = yvk-yvk_old;
                 wuk = Fyuk-Fyuk_old;
                 wvk = Fyvk-Fyvk_old;
                 NormFyk2=Fyuk'*Fyuk+Fyvk'*Fyvk;
                 NormFyk=sqrt(NormFyk2);
                 Normsk2=suk'*suk+svk'*svk;
                 Normsk=sqrt(Normsk2);
                 wktsk=wuk'*suk+wvk'*svk; 
                 lambdak=1+max(0,wktsk/(NormFyk)*(Normsk)^2);
                 yk1u=wuk+lambdak*norm(Fyuk)*suk;
                 yk1v=wvk+lambdak*norm(Fyvk)*svk;
                 dktyk1=dku'*yk1u+dkv'*yk1v;
%                  Fyktwk = Fyuk'*wuk+Fyvk'*wvk;
                Fyktdk = Fyuk'*dku+Fyvk'*dkv;
                Fyktyk1= Fyuk'*yk1u+Fyvk'*yk1v;
               Normwk2 = wuk'*wuk+wvk'*wvk;
                Normwk = sqrt(Normwk2);
                Normyk12=yk1u'*yk1u+yk1v'*yk1v;
                Normyk1=sqrt(Normyk12);
               tk=min(0.3,max(0,1-wktsk/Normwk^2)); % yk1=yk+0.01*dk    ֵЧ  û  ֱ  ʹ  ykЧ    
                 mu=0.2;   
                 fenmu=max(norm(NormFyk2_old,max(mu*Normdk*Normyk1,dktyk1))); 
                 thetak=tk*Fyktdk/fenmu;  % ԭʼ ķ ĸ   norm(Fk0)^2  
                 betak=Fyktyk1/fenmu-Normyk1^2*Fyktdk/fenmu^2; %(max(mu*norm(dk)*norm(Fk),norm(Fk0)^2))
                 dku=-Fyuk+betak*dku+thetak*yk1u;
                 dkv = -Fyvk+betak*dkv+thetak*yk1v;     
             case'MITTCGP'
                 wuk = Fyuk-Fyuk_old;
                 wvk = Fyvk-Fyvk_old;
                 dktwk=dku'*wuk+dkv'*wvk;
                Normdk2=dku'*dku+dkv'*dkv;
                Normdk=sqrt(Normdk2);
                 lambdak=1+max(0,(-1)*dktwk/Normdk2);
                 yk1u=wuk+lambdak*dku;
                 yk1v=wvk+lambdak*dkv;
                 dktyk1=dku'*yk1u+dkv'*yk1v;
                Fyktdk = Fyuk'*dku+Fyvk'*dkv;
                Fyktyk1= Fyuk'*yk1u+Fyvk'*yk1v;
                 mu=0.1;   
               Normyk12=yk1u'*yk1u+yk1v'*yk1v;
                Normyk1=sqrt(Normyk12);
%                 tk=0.4;
                fenmu=max(NormFyk2_old,max(mu*Normdk*Normyk1,dktyk1)); 
                 thetak=0.4*Fyktdk/fenmu;  % ԭʼ ķ ĸ   norm(Fk0)^2  
                 betak=Fyktyk1/fenmu-Normyk1^2*Fyktdk/fenmu^2; %(max(mu*norm(dk)*norm(Fk),norm(Fk0)^2))
                 dku=-Fyuk+betak*dku+thetak*yk1u;
                 dkv = -Fyvk+betak*dkv+thetak*yk1v;
             case'ITHCGPM'
                  tau=0.2;
                 wuk = Fyuk-Fyuk_old;
                 wvk = Fyvk-Fyvk_old;
                 dktwk=dku'*wuk+dkv'*wvk;
               pku=Fyuk-norm(Fyuk)/norm(Fyuk_old)*Fyuk_old;
                pkv=Fyvk-norm(Fyvk)/norm(Fyvk_old)*Fyvk_old;  
                 vk=tau*(Normdk^2+NormFyk^2)+max(NormFyk2_old,dktwk);
                 Fyktpk=Fyuk'*pku+Fyvk'*pkv;
                Fyktdk = Fyuk'*dku+Fyvk'*dkv;
                 Normpk2 = pku'*pku+pkv'*pkv;
                Normpk = sqrt(Normpk2);
                betak=Fyktpk/vk-Normpk^2*Fyktdk/vk^2;
                thetak=0.1*Fyktdk\vk; 
                 dku=-Fyuk+betak*dku+thetak*pku;
                 dkv = -Fyvk+betak*dkv+thetak*pkv;
             case'FITTCGPM-PRP'
                 wuk = Fyuk-Fyuk_old;
                 wvk = Fyvk-Fyvk_old;
                 NormFyk2=Fyuk'*Fyuk+Fyvk'*Fyvk;
                 NormFyk=sqrt(NormFyk2);
                 NormFyk2_old = Fyuk_old'*Fyuk_old+Fyvk_old'*Fyvk_old;
                NormFyk_old = sqrt(NormFyk2_old);
                Normdk2=dku'*dku+dkv'*dkv;
                Normdk=sqrt(Normdk2);
                Fyktwk= Fyuk'*wuk+Fyvk'*wvk;
                betak=Fyktwk/NormFyk2_old;
                FyktFyk_old=Fyuk'*Fyuk_old+Fyvk'*Fyvk_old;
                thetak=-0.055*FyktFyk_old/NormFyk2_old;%-0.34545
                muk=0.001*NormFyk/max(NormFyk_old,(abs(betak))*Normdk);
                 dku=-Fyuk+muk*betak*dku+thetak*Fyuk_old;
                 dkv=-Fyvk+muk*betak*dkv+thetak*Fyvk_old;      
                 case'FITTCGPM-DY'
                 wuk = Fyuk-Fyuk_old;
                 wvk = Fyvk-Fyvk_old;
                 NormFyk2=Fyuk'*Fyuk+Fyvk'*Fyvk;
                 NormFyk=sqrt(NormFyk2);
                 NormFyk2_old = Fyuk_old'*Fyuk_old+Fyvk_old'*Fyvk_old;
                NormFyk_old = sqrt(NormFyk2_old);
                Normdk2=dku'*dku+dkv'*dkv;
                Normdk=sqrt(Normdk2);
%                 Fyktwk= Fyuk'*wuk+Fyvk'*wvk;
                dktwk=dku'*wuk+dkv'*wvk;
                betak=NormFyk2_old/dktwk;
                FyktFyk_old=Fyuk'*Fyuk_old+Fyvk'*Fyvk_old;
                thetak=-0.055*FyktFyk_old/NormFyk2_old;%-0.34545
                muk=0.001*NormFyk/max(NormFyk_old,abs(betak)*Normdk);
                 dku=-Fyuk+muk*betak*dku+thetak*Fyuk_old;
                 dkv=-Fyvk+muk*betak*dkv+thetak*Fyvk_old;     
            case'ISADFM'%2021  Accelerated derivative-free method for nonlinear monotone equations with an application Abdulkarim Hassan Ibrahim1 Poom Kumam1,2,3Auwal Bala Abubakar4,5 Abubakar Adamu2,6
                 uku=Fzuk_new-Fyuk_old;
                 ukv=Fzvk_new-Fyvk_old;
                 Fyktdk = Fyuk'*dku+Fyvk'*dkv;
                 dktuk=dku'*uku+dkv'*ukv;
                 Fyktuk=Fyuk'*uku+Fyvk'*ukv;
                 thetak=Fyktdk/dktuk;  % ԭʼ ķ ĸ   norm(Fk0)^2  
                 betak=Fyktuk/dktuk; %(max(mu*norm(dk)*norm(Fk),norm(Fk0)^2))
                 dku=-Fyuk+betak*dku+thetak*uku;  
                 dkv=-Fyvk+betak*dkv+thetak*ukv;
             case'ISTCP'%Ibrahim, A.H., Kumam, P ., Sun, M., Chaipunya, P .: Projection method with inertial step for nonlinear equations: Application to signal recovery. J. Ind. Manag. Optim. 19(1), 30 C55 (2023)
                FyktFyk_old = Fyuk'*Fyuk_old+Fyvk'*Fyvk_old;
                Fyktdk = Fyuk'*dku+Fyvk'*dkv;
                Fyk_oldtdk = Fyuk_old'*dku+Fyvk_old'*dkv;
                betak = FyktFyk_old/(Fyk_oldtdk);
                thetak = Fyktdk/(Fyk_oldtdk); 
                dku = -Fyuk+betak*dku+thetak*Fyuk_old;
                dkv = -Fyvk+betak*dkv+thetak*Fyvk_old;  
        end
    end
    Normdk2 = dku'*dku+dkv'*dkv;
    Normdk = sqrt(Normdk2);
    if Normdk<epsilon1
        L1 = 1;
        NormF = NormFyk;
        mses(k) = 1/n*norm(yk-true_x)^2;
        Tcpu(k) = cputime-t0;
        break;
    end
    
    %%% Start Armijo-type line search  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % model=1 means -F(zk)'*dk    sigma*tk*norm(dk)^2
    % model=2 means -F(zk)'*dk    sigma*tk*norm(F(zk))*norm(dk)^2
    % model=3 means -F(zk)'*dk    sigma*tk*norm(F(zk))/(1+norm(F(zk)))*norm(dk)^2
    % model=4 means -F(zk)'*dk    sigma*tk*min(nu,norm(Fz_new,2))*norm(dk)^2
    if model==1
        t = gamma;
        zuk_new = yuk+t*dku;
        zvk_new = yvk+t*dkv;
        zk_new = zuk_new-zvk_new;
        Azk_new = A(zk_new);
        tempz = AT(Azk_new)-Atb;
        Fzuk_new = min(zuk_new,tempz+lambda);
        Fzvk_new = min(zvk_new,-tempz+lambda);
        NF = NF+1;
        Fzk_newtdk = Fzuk_new'*dku+Fzvk_new'*dkv;
        % check the Armijo-type line search condition
        while -Fzk_newtdk < sigma*t*Normdk2 && t>10^-6  
            % the Armijo-type line search condition violated
            t = t*tau;
            zuk_new = yuk+t*dku;
            zvk_new = yvk+t*dkv;
            zk_new = zuk_new-zvk_new;
            Azk_new = A(zk_new);
            tempz = AT(Azk_new)-Atb;
            Fzuk_new = min(zuk_new,tempz+lambda);
            Fzvk_new = min(zvk_new,-tempz+lambda);
            NF = NF+1;
            Fzk_newtdk = Fzuk_new'*dku+Fzvk_new'*dkv;
        end %%% End Armijo-type line search %%%
        NormFzk_new2 = Fzuk_new'*Fzuk_new+Fzvk_new'*Fzvk_new;
        NormFzk_new = sqrt(NormFzk_new2);
    elseif model==2
        t = gamma;
        zuk_new = yuk+t*dku;
        zvk_new = yvk+t*dkv;
        zk_new = zuk_new-zvk_new;
        Azk_new = A(zk_new);
        tempz = AT(Azk_new)-Atb;
        Fzuk_new = min(zuk_new,tempz+lambda);
        Fzvk_new = min(zvk_new,-tempz+lambda);
        NF = NF+1;
        Fzk_newtdk = Fzuk_new'*dku+Fzvk_new'*dkv;
        NormFzk_new2 = Fzuk_new'*Fzuk_new+Fzvk_new'*Fzvk_new;
        NormFzk_new = sqrt(NormFzk_new2);
        % check the Armijo-type line search condition
        while -Fzk_newtdk < sigma*t*NormFzk_new*Normdk2 && t>10^-6  
            % the Armijo-type line search condition violated
            t = t*tau;
            zuk_new = yuk+t*dku;
            zvk_new = yvk+t*dkv;
            zk_new = zuk_new-zvk_new;
            Azk_new = A(zk_new);
            tempz = AT(Azk_new)-Atb;
            Fzuk_new = min(zuk_new,tempz+lambda);
            Fzvk_new = min(zvk_new,-tempz+lambda);
            NF = NF+1;
            Fzk_newtdk = Fzuk_new'*dku+Fzvk_new'*dkv;
            NormFzk_new2 = Fzuk_new'*Fzuk_new+Fzvk_new'*Fzvk_new;
            NormFzk_new = sqrt(NormFzk_new2);
        end %%% End Armijo-type line search %%%
    elseif model==3
        t = gamma;
        zuk_new = yuk+t*dku;
        zvk_new = yvk+t*dkv;
        zk_new = zuk_new-zvk_new;
        Azk_new = A(zk_new);
        tempz = AT(Azk_new)-Atb;
        Fzuk_new = min(zuk_new,tempz+lambda);
        Fzvk_new = min(zvk_new,-tempz+lambda);
        NF = NF+1;
        Fzk_newtdk = Fzuk_new'*dku+Fzvk_new'*dkv;
        NormFzk_new2 = Fzuk_new'*Fzuk_new+Fzvk_new'*Fzvk_new;
        NormFzk_new = sqrt(NormFzk_new2);
        % check the Armijo-type line search condition
        while -Fzk_newtdk < sigma*t*NormFzk_new/(1+NormFzk_new)*Normdk2 && t>10^-6  
            % the Armijo-type line search condition violated
            t = t*tau;
            zuk_new = yuk+t*dku;
            zvk_new = yvk+t*dkv;
            zk_new = zuk_new-zvk_new;
            Azk_new = A(zk_new);
            tempz = AT(Azk_new)-Atb;
            Fzuk_new = min(zuk_new,tempz+lambda);
            Fzvk_new = min(zvk_new,-tempz+lambda);
            NF = NF+1;
            Fzk_newtdk = Fzuk_new'*dku+Fzvk_new'*dkv;
            NormFzk_new2 = Fzuk_new'*Fzuk_new+Fzvk_new'*Fzvk_new;
            NormFzk_new = sqrt(NormFzk_new2);
        end %%% End Armijo-type line search %%%
     elseif model==4
        t = gamma;
        zuk_new = yuk+t*dku;
        zvk_new = yvk+t*dkv;
        zk_new = zuk_new-zvk_new;
        Azk_new = A(zk_new);
        tempz = AT(Azk_new)-Atb;
        Fzuk_new = min(zuk_new,tempz+lambda);
        Fzvk_new = min(zvk_new,-tempz+lambda);
        NF = NF+1;
        Fzk_newtdk = Fzuk_new'*dku+Fzvk_new'*dkv;
        NormFzk_new2 = Fzuk_new'*Fzuk_new+Fzvk_new'*Fzvk_new;
        NormFzk_new = sqrt(NormFzk_new2);
        eta_k=max(0.001, min(0.8,NormFzk_new2)); % check the Armijo-type line search condition
%                  while -Fz new'*dk < sigma*t*eta_k*norm(dk,2)^2 && 10^(-10)
        % check the Armijo-type line search condition
        while -Fzk_newtdk < sigma*t*eta_k*Normdk2 && t>10^(-6)  
            % the Armijo-type line search condition violated
            t = t*tau;
            zuk_new = yuk+t*dku;
            zvk_new = yvk+t*dkv;
            zk_new = zuk_new-zvk_new;
            Azk_new = A(zk_new);
            tempz = AT(Azk_new)-Atb;
            Fzuk_new = min(zuk_new,tempz+lambda);
            Fzvk_new = min(zvk_new,-tempz+lambda);
            NF = NF+1;
            Fzk_newtdk = Fzuk_new'*dku+Fzvk_new'*dkv;
            NormFzk_new2 = Fzuk_new'*Fzuk_new+Fzvk_new'*Fzvk_new;
            NormFzk_new = sqrt(NormFzk_new2);
        end
    else
        nu = 1;
        t = gamma;
        zuk_new = yuk+t*dku;
        zvk_new = yvk+t*dkv;
        zk_new = zuk_new-zvk_new;
        Azk_new = A(zk_new);
        tempz = AT(Azk_new)-Atb;
        Fzuk_new = min(zuk_new,tempz+lambda);
        Fzvk_new = min(zvk_new,-tempz+lambda);
        NF = NF+1;
        Fzk_newtdk = Fzuk_new'*dku+Fzvk_new'*dkv;
        NormFzk_new2 = Fzuk_new'*Fzuk_new+Fzvk_new'*Fzvk_new;
        NormFzk_new = sqrt(NormFzk_new2);
        % check the Armijo-type line search condition
        while -Fzk_newtdk < sigma*t*min(nu,NormFzk_new)*Normdk2 && t>10^-6  
            % the Armijo-type line search condition violated
            t = t*tau;
            zuk_new = yuk+t*dku;
            zvk_new = yvk+t*dkv;
            zk_new = zuk_new-zvk_new;
            Azk_new = A(zk_new);
            tempz = AT(Azk_new)-Atb;
            Fzuk_new = min(zuk_new,tempz+lambda);
            Fzvk_new = min(zvk_new,-tempz+lambda);
            NF = NF+1;
            Fzk_newtdk = Fzuk_new'*dku+Fzvk_new'*dkv;
            NormFzk_new2 = Fzuk_new'*Fzuk_new+Fzvk_new'*Fzvk_new;
            NormFzk_new = sqrt(NormFzk_new2);
        end %%% End Armijo-type line search %%%
    end
    zuk = zuk_new;
    zvk = zvk_new;
    zk = zk_new;
    Fzuk = Fzuk_new;
    Fzvk = Fzvk_new;
    NormFzk2 = NormFzk_new2;
    NormFzk = NormFzk_new;
    if NormFzk<=epsilon
        L1 = 1;
        NormF = NormFzk; % the final norm of equations
        mses(k) = 1/n*norm(zk-true_x)^2;
        Tcpu(k) = cputime-t0;
        break;
    end
    Fzktykzk = Fzuk'*(yuk-zuk)+Fzvk'*(yvk-zvk);
    xik = Fzktykzk/NormFzk2;
    % compute the next iteration 
    xu1 = yuk-rho*xik*Fzuk;
    xv1 = yvk-rho*xik*Fzvk;
%     xuv1min = min(xu1,xv1);
%     xu1 = xu1-xuv1min;
%     xv1 = xv1-xuv1min;
    x1 = xu1-xv1;
    mses(k) = 1/n*norm(x1-true_x)^2;
    Ax1 = A(x1);
    tempx = AT(Ax1)-Atb;
    Fxu = min(xu1,tempx+lambda);
    Fxv = min(xv1,-tempx+lambda);
    NF = NF+1;
    NormFx2 = Fxu'*Fxu+Fxv'*Fxv;
    NormFx = sqrt(NormFx2);
    if NormFx<=epsilon
        L1 = 1;
        NormF = NormFx;
        mses(k) = 1/n*norm(x1-true_x)^2;
        Tcpu(k) = cputime-t0;
        break;
    end
    
    % update the iteration
    xu0_old = xu0;
    xv0_old = xv0;
    xu0 = xu1;
    xv0 = xv1;
    yuk_old = yuk;
    yvk_old = yvk;
    Fyuk_old = Fyuk;
    Fyvk_old = Fyvk;
    NormFyk2_old = NormFyk2;
    Tcpu(k) = cputime-t0;
end
if L1~=1
    NF = NaN;
    NormF = NaN;
end
