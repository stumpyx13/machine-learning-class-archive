% Problem 4 for PSET 1 for 6.867 Machine Learning Fall 2017

%% Options
plotWeights                 =   0;
plotRegression              =   1;
%% Load data
[x_T, y_T]                  =   lassoTestData();
[x_Tr, y_Tr]                =   lassoTrainData();
[x_V, y_V]                  =   lassoValData();

data_true                   =   importdata('lasso_true_w.txt');
w_true                      =   data_true(1,:);

%% Transform data into appropriate basis
m                           =   12;
PHI_T                       =   sinBasis(x_T,m);
PHI_Tr                      =   sinBasis(x_Tr,m);
PHI_V                       =   sinBasis(x_V,m);

%% parameters
lambda                      =   0.05;
1;

[w_lasso, stats]            =   lasso(PHI_Tr,y_Tr','Lambda',lambda);
w_ridge                     =   ridgeRegress(PHI_Tr,y_Tr',lambda);

if(plotWeights)
    hold on;
    figure(1);
    plot(abs(w_true),'-*')
    plot(abs(w_lasso),'-*')
    plot(abs(w_ridge),'-*')

    xlabel('Weight vector entry number','FontSize',16);
    ylabel('Magnitude','FontSize',16);
    set(gca,'FontSize',16);
end

if(plotRegression)
   figure(2);
   hold on
   x_all                    =   linspace(min([min(x_T),min(x_Tr),min(x_V)]),max([max(x_T),max(x_Tr),max(x_V)]),1000);
   PHI_all                  =   sinBasis(x_all,m);
   y_all_lasso              =   PHI_all * w_lasso;
   y_all_ridge              =   PHI_all * w_ridge;
   y_all_true               =   PHI_all * w_true';
   scatter(x_T,y_T)
   scatter(x_Tr,y_Tr)
   scatter(x_V,y_V);
   plot(x_all,y_all_true)
   xlabel('x','FontSize',16);
   ylabel('y','FontSize',16);
   set(gca,'FontSize',16);
   plot(x_all,y_all_lasso)
   plot(x_all,y_all_ridge)
end