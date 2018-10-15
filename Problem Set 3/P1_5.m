% 6.867 PSET 3 Problem 1.5 MNIST data classification with NN
clear;

%% Options
normalizeData                   =   1;
showMisclassifiedDigit          =   1;
N_hiddenLayers                      =   2;
N_layerSize                         =   500;
max_epoch                           =   500;
eta0                                =   0.0001;
mu                                  =   0.9;


%% Specifications

mnist_digits_class1             =   [0,1,2,3,4,5,6,7,8,9];


N_train                         =   200;
N_val                           =   150;
N_test                          =   150;

N_vector                        =   784;

baseFile                        =   'hw3_resources/data/mnist_digit_';
extension                       =   '.csv';
%% Initialization

X_train                         =   zeros(N_train*(length(mnist_digits_class1)),N_vector);
X_val                           =   zeros(N_val*(length(mnist_digits_class1)) ,N_vector);
X_test                          =   zeros(N_test*(length(mnist_digits_class1)),N_vector);

Y_train                         =   zeros(N_train*(length(mnist_digits_class1)),1);
Y_val                           =   zeros(N_val*(length(mnist_digits_class1)),1);
Y_test                          =   zeros(N_test*(length(mnist_digits_class1)),1);

%% Read in data

for i = 1:length(mnist_digits_class1)
   filename                     =   [baseFile, num2str(mnist_digits_class1(i)), extension];
   fileID                       =   fopen(filename);
   A                            =   cell2mat(textscan(fileID, repmat('%f',[1,N_vector])));
   X_train((i-1)*N_train +1:i*N_train,:)    =   A(1:N_train,:);
   X_val((i-1)*N_val +1:i*N_val,:)          =   A((N_train+1):(N_train+N_val),:);
   X_test((i-1)*N_test +1:i*N_test,:)       =   A((N_train+N_val+1):(N_train+N_val+N_test),:);
   
   Y_train((i-1)*N_train +1:i*N_train,1)    =   mnist_digits_class1(i)+1;
   Y_val((i-1)*N_val +1:i*N_val,1)          =   mnist_digits_class1(i)+1;
   Y_test((i-1)*N_test +1:i*N_test,1)       =   mnist_digits_class1(i)+1;
   fclose(fileID);
end


%% Normalize data
if(normalizeData == 1)
    X_train                             =   (2*X_train)./(255) - 1;
    X_val                               =   (2*X_val)./(255) - 1;
    X_test                              =   (2*X_test)./(255) - 1;
end

%% Train NN
NN                                  =   trainNeuralNetworkClassification(X_train,Y_train,N_hiddenLayers,...
                                            ones(1,N_hiddenLayers)*N_layerSize, 10,...
                                            max_epoch,eta0,mu, [Y_val,X_val]);
                                        
%% Test set error
Y_NN_test                           =   NN_classify(NN,X_test);
N_errors                            =   sum(Y_NN_test ~= Y_test);
disp(['Neural Network parameters: ', num2str(N_hiddenLayers), ' hidden layers, ', num2str(N_layerSize),' neurons per layer']);
disp(['Number of errors: ', num2str(N_errors), ' (', num2str(N_errors/length(Y_test) * 100), ' %)']);