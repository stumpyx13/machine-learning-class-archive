% Problem 1.1 for 6.867 PSET 1, September 2017

%% Load data

global A b u covMat

[gaussMean, gaussCov, quadBowlA, quadBowlb] = loadParametersP1();

A                                           =   quadBowlA;
b                                           =   quadBowlb;
u                                           =   gaussMean;
covMat                                      =   gaussCov;

%% Options
investigateNegGauss                         =   0;
investigateQuadBowl                         =   1;

%% Set true minimum

x_opt_Gauss                                 =   gaussMean;
x_opt_quadBowl                              =   quadBowlA\quadBowlb;

[n_g,m_g]                                   =   size(gaussMean);
[n_qb,m_qb]                                 =   size(quadBowlb);

%% Set Gradient descent parameters

method                                      =   'SteepestDescent';
stepSizeMethod                              =   'Constant';
errType                                     =   'iterNorm';

alpha0                                      =   0.05;

x_0_Gauss                                   =   zeros(n_g,m_g);
x_0_quadBowl                                =   zeros(n_g,m_g);
x_0_quadBowl(1)                             =   -500;

eps                                         =   1e-6;

%% Run Gradient descent

if(investigateNegGauss == 1)
    [x_opt_gd_Gauss, errArray_Gauss, x_iterArray_Gauss]         =   gradientDescent(x_0_Gauss,...
                        @negativeGaussFunction, @gradNegativeGaussFunct, [], method, stepSizeMethod,...
                        errType, eps, alpha0, x_opt_Gauss, 10000);
    hold on;
    figure(1);
    iter                    =   1:length(errArray_Gauss);
    plot(iter,errArray_Gauss);   
    set(gca,'FontSize',16);
    xlabel('Iteration Number','FontSize',16);
    ylabel('||x_k - x^*||','FontSize',18)
    title(['Iteration norm stopping criteria, \epsilon = ',num2str(eps)]);
end


if(investigateQuadBowl == 1)
    [x_opt_gd_quadBowl, errArray_quadBowl, x_iterArray_quadBowl]         =   gradientDescent(x_0_quadBowl,...
                        @quadraticBowlFunct, @quadbowlGrad, [], method, stepSizeMethod,...
                        errType, eps, alpha0, x_opt_quadBowl, 1000);
    hold on;
    figure(2);
    iter                    =   1:length(errArray_quadBowl);
    plot(iter,errArray_quadBowl);
    set(gca,'FontSize',16);
    xlabel('Iteration Number','FontSize',16);
    ylabel('||x_k - x^*||','FontSize',18)
    title(['Iteration norm stopping criteria, \epsilon = ',num2str(eps)]);
    
end



