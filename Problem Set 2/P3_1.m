% 6.867 PSET 2 Problem 3.1

%% Set global variables
clear;
global X Y kernal K_gram lambda_RBF

kernal          =   @linearKernal;
lambda_RBF      =   1;

%% Load data
dataSetN                    =   3;
dataFilePath                =   ['hw2_resources/data/data',num2str(dataSetN),'_train.csv'];
fileID                      =   fopen(dataFilePath);
data                        =   textscan(fileID,'%f %f %f');
data                        =   cell2mat(data);
fclose(fileID);

X                           =   data(:,1:2);
Y                           =   data(:,3);

%% SVM parameters
K_gram          =   svm_augmentGram();
lambda          =   0.02;
epochs          =   1000;

%% Train SVM

w               =   pegasosLearningAlg(lambda, epochs);

%% Plot data
hold on
[xx,yy,zz]      =   plotDecisionBoundary_base(X,Y,w,@linearClassifierValue);

[C,h]=contour(xx, yy, zz, [0,0],'k','LineWidth',2);
set(h,'ShowText','off');
[C,h]=contour(xx, yy, zz, [-1,-1],'k--','LineWidth',2);
set(h,'ShowText','off');

scatter(X(Y==1,1),X(Y==1,2),40,'bo');
scatter(X(Y==-1,1),X(Y==-1,2),40,'ro');
[C,h]=contour(xx, yy, zz, [1,1],'k--','LineWidth',2);
set(h,'ShowText','off');
set(gca,'FontSize',16);
xlabel('x_1','FontSize',18);
ylabel('x_2','FontSize',18);
legend('Decision Boundary','Margins','+1 values','-1 values');
title(['data set ',num2str(dataSetN),' training']);

w_act           =   w(2:end);
margin_width    =   1/sqrt(w_act'*w_act)