% 6.867 PSET 2 Problem 1.2
clear;
%% Load data
dataSetN                    =   4;
dataFilePath                =   ['hw2_resources/data/data',num2str(dataSetN),'_train.csv'];
fileID                      =   fopen(dataFilePath);
data                        =   textscan(fileID,'%f %f %f');
data                        =   cell2mat(data);
fclose(fileID);

dataFileValidate            =   ['hw2_resources/data/data',num2str(dataSetN),'_validate.csv'];
fileID                      =   fopen(dataFileValidate);
data_val                    =   textscan(fileID,'%f %f %f');
data_val                    =   cell2mat(data_val);
fclose(fileID);

dataFileTest                =   ['hw2_resources/data/data',num2str(dataSetN),'_test.csv'];
fileID                      =   fopen(dataFileTest);
data_test                   =   textscan(fileID,'%f %f %f');
data_test                   =   cell2mat(data_test);
fclose(fileID);

X_val                       =   data_val(:,1:2);
Y_val                       =   data_val(:,3);

X_test                      =   data_test(:,1:2);
Y_test                      =   data_test(:,3);

global X Y lambda

X                           =   data(:,1:2);

Y                           =   data(:,3);
lambdaArray                 =   logspace(-3,2,50);

w_L1                        =   zeros(3,length(lambdaArray));
w_L2                        =   zeros(3,length(lambdaArray));
misclassifyRate_L1          =   zeros(1,length(lambdaArray));
misclassifyRate_L2          =   zeros(1,length(lambdaArray));

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
    %% Calculate misclassification rate

    Y_val_L1           =   zeros(size(Y_val));
    Y_val_L2           =   zeros(size(Y_val));

    for j = 1:length(Y_val_L1)
       Y_val_L1(j)     =   classify_l1_logreg(X_val(j,:)',w_L1(:,i));
       Y_val_L2(j)     =   logist_classify(X_val(j,:)',w_L2(:,i)); 
    end


    Y_val_L2(Y_val_L2 > 0.5)      =   1;
    Y_val_L2(Y_val_L2 <= 0.5)     =   -1;

    sim_L1                          =   Y_val == Y_val_L1;
    sim_L2                          =   Y_val == Y_val_L2;

    N_misclassify_L1                =   length(Y_val) - sum(sim_L1);
    N_misclassify_L2                =   length(Y_val) - sum(sim_L2);

    misclassifyRate_L1(i)              =   N_misclassify_L1/length(Y_val) * 100;
    misclassifyRate_L2(i)              =   N_misclassify_L2/length(Y_val) * 100;
end

h = figure(1);
semilogx(lambdaArray,misclassifyRate_L1,lambdaArray,misclassifyRate_L2);
set(gca,'FontSize',16);
xlabel('\lambda','FontSize',18);
ylabel('Misclassification Rate (%)','FontSize',16);
title(['data set: ',num2str(dataSetN)]);

[minE_L1, ind_minE_L1]              =   min(misclassifyRate_L1);
[minE_L2, ind_minE_L2]              =   min(misclassifyRate_L2);
w_L1_opt                            =   w_L1(:,ind_minE_L1);
w_L2_opt                            =   w_L2(:,ind_minE_L2);

disp(['Minimum error for L_1: ',num2str(minE_L1),'% (\lambda = ',...
    num2str(lambdaArray(ind_minE_L1)),')'])
disp(['Minimum error for L_2: ',num2str(minE_L2),'% (\lambda = ',...
    num2str(lambdaArray(ind_minE_L2)),')'])

%% Calculate misclassification rate on test data

Y_test_L1           =   zeros(size(Y_test));
Y_test_L2           =   zeros(size(Y_test));


for j = 1:length(Y_test_L1)
   Y_test_L1(j)     =   classify_l1_logreg(X_test(j,:)',w_L1_opt);
   Y_test_L2(j)     =   logist_classify(X_test(j,:)',w_L2_opt); 
end


Y_test_L2(Y_test_L2 > 0.5)      =   1;
Y_test_L2(Y_test_L2 <= 0.5)     =   -1;

sim_L1_test                          =   Y_test == Y_test_L1;
sim_L2_test                          =   Y_test == Y_test_L2;

N_misclassify_L1                =   length(Y_test) - sum(sim_L1_test);
N_misclassify_L2                =   length(Y_test) - sum(sim_L2_test);

misclassifyRate_L1_test              =   N_misclassify_L1/length(Y_test) * 100;
misclassifyRate_L2_test              =   N_misclassify_L2/length(Y_test) * 100;

disp(['Error on test set for L_1: ',num2str(misclassifyRate_L1_test),'% (\lambda = ',...
    num2str(lambdaArray(ind_minE_L1)),')'])
disp(['Error on test set for L_2: ',num2str(misclassifyRate_L2_test),'% (\lambda = ',...
    num2str(lambdaArray(ind_minE_L2)),')'])

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

