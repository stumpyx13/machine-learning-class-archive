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