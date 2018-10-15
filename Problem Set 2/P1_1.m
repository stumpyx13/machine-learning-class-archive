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
stepSizeOption              =   'Constant';
errType                     =   'gradientNorm';
descentType                 =   'SteepestDescent';
alpha0                      =   5;
maxIter                     =   10000;

w0                          =   [0;1;1];

funct                       =   @logitError_L2Regularizer;
grad                        =   @grad_logitError_L2Regularizer;

%% Run gradient descent

[w_opt, errArray, w_iter_array]           =   gradientDescent(w0, funct, [], [], descentType, stepSizeOption, ...
                                    errType, eps, alpha0, [], maxIter);
                                

 %% Post process
 
[n,m]                       =   size(w_iter_array);
norm_w                      =   zeros(1,m);


for i = 1:m
   norm_w(i)                =   norm(w_iter_array(:,i),2); 
end

plot(1:m, norm_w, '-*');
xlabel('Iteration Number','FontSize',16);
ylabel('||w_k||','FontSize',16);
xlim([1 20]);
set(gca,'FontSize',16);
grid on