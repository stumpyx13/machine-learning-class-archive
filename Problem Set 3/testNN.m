%Script to test NN implementation
datapath                    =   'hw3_resources/data/data_3class_mod.csv';
data                        =   csvread(datapath,1,1);

data_train                  =   data(1:600,:);
data_val                    =   data(601:end,:);

X                           =   data_train(:,1:end-1);
Y                           =   data_train(:,end);

X_val                       =   data_val(:,1:end-1);
Y_val                       =   data_val(:,end);
Y_val                       =   Y_val + 1;

N_classes                   =   3;
Y                           =   Y+1;
N_layers                    =   2;

N_layerSize                 =   50;
N_neurons                   =   N_layerSize*ones(1,N_layers);
max_epoch                   =   500;
eta0                        =   0.0001;
mu                          =   0.9;

NN                          =   trainNeuralNetworkClassification(X,Y, N_layers, N_neurons, N_classes, max_epoch, ...
                                    eta0,mu, [Y_val,X_val]);
                                
y_test                      =   NN_classify(NN,X);

N_errors                    =   sum(Y ~= y_test);
disp(['Neural Network parameters: ', num2str(N_layers), ' hidden layers, ', num2str(N_layerSize),' neurons per layer']);
disp(['Number of errors: ', num2str(N_errors), ' (', num2str(N_errors/length(Y) * 100), ' %)']);