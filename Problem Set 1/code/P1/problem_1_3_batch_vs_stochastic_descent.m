% Problem 1.3 PSET 1 for 6.867 Machine Learning Fall 2017

%% Load data (Note: no need to augment X in this case with column of ones)
global X Y loops calls

[X, Y]                      =   loadFittingDataP1();

[n,m]                       =   size(X);

%% Run batch gradient descent on data for linear regression

theta_analytic              =   (X'*X)\(X'*Y);

theta0                      =   zeros(m,1);
descentType                 =   'SteepestDescent';
stepSizeType                =   'Constant';
eps                         =   1e-3;
maxIter                     =   30000;
alpha0_BGD                      =   0.00001;

[theta_BGD, errArray_BGD, theta_iterArray_BGD]              =   gradientDescent(theta0, @leastSquaresError_linearReg,...
                                        @grad_leastSquaresErr_linearReg, [], descentType, stepSizeType, 'gradientNorm', eps, ...
                                        alpha0_BGD, theta_analytic, maxIter);
                                    
%% Run stochastic gradient descent on data for linear regression
alpha0_SGD                  =   100000;
descentType                 =   'Stochastic';
stepSizeType                 =   'Stochastic';
[theta_SGD, errArray_SGD, theta_iterArray_SGD]              =   gradientDescent(theta0, @leastSquaresError_linearReg,...
                                        @gradStochastic_leastSquaresErr_linearReg, [], descentType, stepSizeType, ...
                                        'trueNorm', eps, ...
                                        alpha0_SGD, theta_analytic, maxIter);
                                    
%% Compare BGD and SGD


iter_grad_BGD               =   1:length(errArray_BGD);
iter_grad_BGD               =   iter_grad_BGD * n;
iter_grad_SGD               =   1:length(errArray_SGD);

figure(1);
semilogy(iter_grad_BGD, errArray_BGD, 'b',iter_grad_SGD,errArray_SGD,'r');
set(gca,'FontSize',16);
xlabel('Point-wise gradient evaluations','FontSize',16);
ylabel('||\theta_t - \theta^*||','FontSize',18);
%xlim([0,5000]);
legend('Batch gradient descent','Stochastic gradient descent');
