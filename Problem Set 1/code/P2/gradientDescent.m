function [ x_opt, errArray, x_iterArray ] = gradientDescent( x0, funct, grad, hess, descentType, ...
    stepSizeOption, errType, eps, alpha0, x_opt_real, maxIter)
%gradientDescent Implements gradient descent 
%   Uses a gradient descent method to calculate the optimum of function funct with gradient grad, grad and
%   funct must be matlab functions that take as input some vector x
%
%   Inputs:
%
%   Outputs:

%% Initialize variables
global calls

calls                   =   0;

x_k1                    =   x0;
err                     =   1;
alpha                   =   1;

% Armijo rule constants
beta                    =   0.5;
sigma                   =   0.3;

% SGD learning rate constants (Robbins-Monro)
tau0                    =   alpha0;
kappa                   =   0.6;
% Set default tolerance
if(isempty(eps))
   eps = 1e-6; 
end

%% while loop for iterations
iter = 1;
ErrArray = [];
x_iterArray(:,1) = x0;
while(err > eps && iter < maxIter)
   
    %Update x_k
    x_k                     =   x_k1;
    
    %Initialize step size
    switch(stepSizeOption)
        case('Armijo')
            alpha           =   1/beta;
        case('Constant')
            alpha           =   alpha0;
    end 
    
    %Calculate gradient
    if(isempty(grad))
        

    else
        grad_k                  =   grad(x_k);
        calls                   =   calls + 1;

    end
    
    %Determine direction d_k
    switch(descentType)
        
        case('SteepestDescent')
            d_k             =   -1;
        case('Newton')
            H               =   hess(x_k);
        case('Stochastic')
            d_k             =   -1;
    end
    
    %Update x_k1
    
    switch(stepSizeOption)
        case('Armijo')
            ar_iterate      =   1;
            f_xk            =   funct(x_k);
            while (ar_iterate)
                alpha       =   alpha * beta;
                x_k1_test   =   x_k + d_k'*grad_k*alpha;
                f_xk1_test  =   funct(x_k1_test);
                LHS_ar      =   f_xk - f_xk1_test;
                RHS_ar      =   -sigma * alpha * grad_k'* (d_k'*grad_k);
            
                ar_iterate  =   LHS_ar < RHS_ar-eps;
            end
            x_k1            =   x_k1_test;
        case('Constant')
            x_k1            =   x_k + d_k'*grad_k*alpha;
        case('Stochastic')
            eta_k           =   (tau0 + iter-1)^-kappa;
            x_k1            =   x_k + d_k*grad_k*eta_k; 
    end
    switch(errType)
        case('gradientNorm')
            err                     =   norm(grad(x_k1));
        case('iterNorm')
            err                     =   norm(x_k1 - x_k);
        case('trueNorm')
            err                     =   norm(x_k1 - x_opt_real);
    end
    if(isempty(x_opt_real))
        errArray(iter)          =   err;
    else
        errArray(iter)          =   norm(x_k1 - x_opt_real);
    end
    iter                    =   iter+1;

    x_iterArray(:,iter)       =   x_k1;
end

x_opt                       =   x_k1;

end

