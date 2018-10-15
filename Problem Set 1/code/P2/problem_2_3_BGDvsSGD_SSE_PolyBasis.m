% Problem 2.3 for PSET 1 for 6.867 Machine Learning Fall 2017
global X Y
%% Load data and set some constants
[x,y]               =   loadFittingDataP2(0);
n                   =   length(y);
d                   =   0.001;
mArray              =   [0, 1, 3, 10];
SSE                 =   zeros(1,length(mArray));
err_grad            =   zeros(1,length(mArray));

Y                   =   y';

eps                 =   1e-4;
maxIter             =   1000000;

%% Options
runBGD              =   1;
runSGD              =   1;

%% Set general gradient descent constants
%beta_0              =   zeros(m+1,1)+1;
BGD_descentType     =   'SteepestDescent';
stepSize_BGD        =   'Constant';
alpha_BGD           =   0.01;
errType             =   'trueNorm';

%% Set stochastic gradient descent constants
SGD_descentType     =   'Stochastic';
stepSize_SGD        =   'Stochastic';
alpha_SGD           =   1;

%% Run for array
iterBGD             =   zeros(1,length(mArray));
iterSGD             =   zeros(1,length(mArray));

for i = 1:length(mArray)
    m                   =   mArray(i);

    beta_anal           =   calcPolyRegCoef(x,y',m);
    beta_0              =   beta_anal-0.5;
    PHI                 =   polyBasis(x,m);
    X                   =   PHI;
    %% Run BGD

    if(runBGD)
        [beta_BGD, errArray_BGD]        =   gradientDescent(beta_0,@sumSquaresError_linearReg,@grad_leastSquaresErr_linearReg,...
                                        [], BGD_descentType, stepSize_BGD, errType, eps, alpha_BGD, beta_anal, maxIter);
        iterBGD(i)                      =   length(errArray_BGD)+1;
        beta_BGD
    end                              
    %% Run SGD

    if(runSGD)
        [beta_SGD, errArray_SGD]    =   gradientDescent(beta_0,@sumSquaresError_linearReg,@gradStochastic_leastSquaresErr_linearReg,...
                                        [], SGD_descentType, stepSize_SGD, errType, eps, alpha_SGD, beta_anal, maxIter);
        iterSGD(i)                      =   length(errArray_SGD)+1;
        beta_SGD
    end

end
fig = figure(1);
set(fig,'Position',[200,200,550,300]);
semilogy(mArray, iterBGD*length(x),'b-*', mArray,iterSGD,'r-*');
set(gca,'FontSize',16);
xlabel('M','FontSize',16);
ylabel('Point-wise gradient iterations','FontSize',16);
legend(['BGD (\alpha = ', num2str(alpha_BGD), ')'],['SGD (\tau_0 = ',num2str(alpha_SGD),' \kappa = 0.6)']);
title(['\epsilon = ',num2str(eps),', ||\theta_{GD} - \theta^*|| <= \epsilon; \theta_0 = \theta^*-0.5'])



