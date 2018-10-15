% 6.867 PSET 2 Problem 1.2
clear;
%% Load data
dataSetN                    =   4;
dataFilePath                =   ['hw2_resources/data/data',num2str(dataSetN),'_train.csv'];
fileID                      =   fopen(dataFilePath);
data                        =   textscan(fileID,'%f %f %f');
data                        =   cell2mat(data);
fclose(fileID);

dataFileValidate            =   ['hw2_resources/data/data',num2str(dataSetN),'_test.csv'];
fileID                      =   fopen(dataFileValidate);
data_val                    =   textscan(fileID,'%f %f %f');
data_val                    =   cell2mat(data_val);
fclose(fileID);

X_val                       =   data_val(:,1:2);
Y_val                       =   data_val(:,3);

global X Y lambda

X                           =   data(:,1:2);

Y                           =   data(:,3);
lambdaArray                 =   [0.01];

w_L1                        =   zeros(3,length(lambdaArray));
w_L2                        =   zeros(3,length(lambdaArray));

%% for loop
for i = 1:length(lambdaArray)
    lambda                      =   lambdaArray(i);


    w0                          =   [0;1;1];

    funct                       =   @logitError_L2Regularizer;
    grad                        =   @grad_logitError_L2Regularizer;

    %% Run LASSO logistic regression

    model_lasso                 =   train_l1_logreg(X,Y,lambda);
    w_L1(:,i)                     =   model_lasso;

    %% Run gradient descent

    %X(:,1)                      =   (X(:,1) - mean(X(:,1)))/std(X(:,1));
    %X(:,2)                      =   (X(:,2) - mean(X(:,2)))/std(X(:,2));
    w_opt                       =   fminunc(@logitError_L2Regularizer,w0);

    w_L2(:,i)                     =   w_opt;
end
%% Plot decision boundary on training data
[xx_L1,yy_L1,zz_L1] = plotDecisionBoundary_LR(X,Y,w_L1,@logist_classify);
[xx_L2,yy_L2,zz_L2] = plotDecisionBoundary_LR(X,Y,w_L2,@logist_classify);

hold on
colormap cool
values              =   [0.5,0.5];
[con,h]             =   contour(xx_L1,yy_L1,zz_L1,values,'k--');

[con2,h2]           =   contour(xx_L2,yy_L2,zz_L2,values,'k');


scatter(X(Y==1,1),X(Y==1,2),'bo');
scatter(X(Y==-1,1),X(Y==-1,2),'ro');

legend('L_1 Decision boundary','L_2 Decision boundary','+1 points','-1 points')
set(gca,'FontSize',16);
xlabel('x_1','FontSize',18);
ylabel('x_2','FontSize',18);

title(['data ',num2str(dataSetN),' training, \lambda = ', num2str(lambda)]);
%% Calculate misclassification rate

Y_test_L1           =   zeros(size(Y_val));
Y_test_L2           =   zeros(size(Y_val));

for i = 1:length(Y_test_L1)
   Y_test_L1(i)     =   classify_l1_logreg(X_val(i,:)',w_L1);
   Y_test_L2(i)     =   logist_classify(X_val(i,:)',w_L2); 
end


Y_test_L2(Y_test_L2 > 0.5)      =   1;
Y_test_L2(Y_test_L2 <= 0.5)     =   -1;

sim_L1                          =   Y_val == Y_test_L1;
sim_L2                          =   Y_val == Y_test_L2;

N_misclassify_L1                =   length(Y_val) - sum(sim_L1);
N_misclassify_L2                =   length(Y_val) - sum(sim_L2);

disp(['Misclassified points for L1: ',num2str(N_misclassify_L1),'(',...
    num2str(N_misclassify_L1/length(Y_val) * 100),'%)'])
disp(['Misclassified points for L2: ',num2str(N_misclassify_L2),'(',...
    num2str(N_misclassify_L2/length(Y_val) * 100),'%)'])

% h = figure(1);
% set(h,'Position',[200,200, 1100, 300])
% subplot(1,3,1)
% semilogx(lambdaArray,w_L1(1,:),'b*-',lambdaArray,w_L2(1,:),'r*-');
% set(gca,'FontSize',16);
% ylabel('||w_{component}||','FontSize',18);
% xlabel('\lambda','FontSize',18);
% title('w_0');
% subplot(1,3,2)
% semilogx(lambdaArray,w_L1(2,:),'b*-',lambdaArray,w_L2(2,:),'r*-');
% set(gca,'FontSize',16);
% xlabel('\lambda','FontSize',18);
% title('w_1');
% subplot(1,3,3)
% semilogx(lambdaArray,w_L1(3,:),'b*-',lambdaArray,w_L2(3,:),'r*-');
% set(gca,'FontSize',16);
% xlabel('\lambda','FontSize',18);
% title('w_0');
% legend('L_1 Regularizer','L_2 Regularizer');

