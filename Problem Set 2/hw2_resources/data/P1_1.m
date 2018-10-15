% 6.867 PSET 2 Problem 1.1

%% Load data
dataFilePath                =   'hw2_resources/data/data1_train.csv';
fileID                      =   fopen(dataFilePath);
data                        =   textscan(fileID,'%f %f %f');
data                        =   cell2mat(data);

global X Y lambda

X                           =   data(:,1:2);
Y                           =   data(:,3);
lambda                      =   0;

%% Set up gradient descent

eps                         =   1e-4;
stepSizeOption              =   'Armijo';
errType                     =   'gradientNorm';
descentType                 =   'SteepestDescent';
alpha0                      =   1;
maxIter                     =   5000;

w0                          =   [0;1;1];

funct                       =   @logitError_L2Regularizer;
grad                        =   [];

%% Run gradient descent

[w_opt, errArray]           =   gradientDescent(w0, funct, grad, [], descentType, stepSizeOption, ...
                                    errType, eps, alpha0, [], maxIter);