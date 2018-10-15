%6.867 PSET3 1.4 - Binary classification
clear;
%% Options
N_set                               =   4;
basePath                            =   'hw3_resources/data/data';
N_hiddenLayers                      =   2;
N_layerSize                         =   5;
max_epoch                           =   500;
eta0                                =   0.0001;
mu                                  =   0.9;

values                              =   [1, 2];

%% Load data
dataPath_train                      =   [basePath,num2str(N_set),'_train_mod.csv'];
dataPath_validate                   =   [basePath,num2str(N_set),'_validate_mod.csv'];
dataPath_test                       =   [basePath,num2str(N_set),'_test_mod.csv'];

data_train                          =   csvread(dataPath_train,1,1);
data_validate                       =   csvread(dataPath_validate,1,1);
data_test                           =   csvread(dataPath_test,1,1);

%% Modify data
X_train                             =   data_train(:,1:end-1);
Y_train                             =   data_train(:,end);

X_validate                          =   data_validate(:,1:end-1);
Y_validate                          =   data_validate(:,end);

X_test                              =   data_test(:,1:end-1);
Y_test                              =   data_test(:,end);

Y_train(Y_train == -1)              =   2;
Y_validate(Y_validate == -1)        =   2;
Y_test(Y_test == -1)                =   2;

%% Train NN
NN                                  =   trainNeuralNetworkClassification(X_train,Y_train,N_hiddenLayers,...
                                            ones(1,N_hiddenLayers)*N_layerSize, 2,...
                                            max_epoch,eta0,mu, [Y_validate,X_validate]);
                                        
%% Test set error
Y_NN_test                           =   NN_classify(NN,X_test);
N_errors                            =   sum(Y_NN_test ~= Y_test);
disp(['Neural Network parameters: ', num2str(N_hiddenLayers), ' hidden layers, ', num2str(N_layerSize),' neurons per layer']);
disp(['Number of errors: ', num2str(N_errors), ' (', num2str(N_errors/length(Y_test) * 100), ' %)']);

%% Plot decision boundaries
[xx, yy, zz]                        =   plotDecisionBoundary_NN(X_train,NN,@NN_classify);

X                                   =   X_train;
Y                                   =   Y_train;
hold on
h                                   =   figure(1);
set(h,'Position',[400,400,800,600])
[C,h]=contour(xx, yy, zz, values,'k','LineWidth',2);
set(h,'ShowText','off');
%Plot the training points
scatter(X(Y==1,1),X(Y==1,2),60,'b.');
scatter(X(Y==2,1),X(Y==2,2),60,'r.');
xlabel('x_1','FontSize',18);
ylabel('y_1','FontSize',18);
set(gca,'FontSize',16);
title(['NN Decision boundary for DS',num2str(N_set),'; Hidden layers = ',num2str(N_hiddenLayers),', ' num2str(N_layerSize),...
            ' units per']);