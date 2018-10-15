function NN = trainNeuralNetworkClassification(X, Y, N_hiddenLayers, N_layerSize, N_classes, max_epoch, eta0, mu, data_Val)
%trainNeuralNetwork train a neural network with the given parameters.
%   Current implementation assumes softmax output layer and ReLU activations
%   X is the input data, with a row corresponding to an observation
%   Y is the output data, with each row corresponding to an observation, should correspond to X
%   N_hiddenLayers is the number of hidden layers for the neural net
%   N_layerSize is a vector with the number of neurons per layer
%   max_epoch is the maximum number of epochs for SGD
%   eta0 is the initial learning rate for SGD
%   mu is the momentum hyperparameter, equal to 0 by default if empty
%   data_Val is the validation set used for early stopping. If the accuracy on the validation set has not decreased
%   for 5 epochs, then stop training

%   General notation:
%       W = weight matrix, corresponds to weights for a specific layer, one column for one node
%       w = weight vector, corresponds to weights for a specific node
%       z = vector of summed inputs for each node in a layer, plus the bias
%       b = vector of bias values for a layer
%       a = vector of activated values for neurons in a layer
%   Neural network parameters for each layer
%   1) Size in neurons
%   2) Activation used
%   3) Weight matrix
%   4) Bias vector
%% Initialization
disp('Initializing Neural Network for Classification');
v_vect                  =   cell(2,N_hiddenLayers+1);
eta                     =   eta0;
N_earlyStop             =   10;

[m_data,n_data]         =   size(X);
v_vect(1,1)             =   {zeros(n_data,N_layerSize(1))};
v_vect(2,1)             =   {zeros(N_layerSize(1),1)};
NN                      =   cell(4,N_hiddenLayers+1);
NN(2,1:end-1)           =   {'ReLU'};
NN(2,end)               =   {'softmax'};
NN(1,end)               =   {N_classes};
NN(1,1)                 =   {N_layerSize(1)};
NN(3,1)                 =   {normrnd(0,sqrt(1/N_layerSize(1)),n_data,N_layerSize(1))};
NN(4,1)                 =   {zeros(N_layerSize(1),1)};
for i = 2:N_hiddenLayers
   NN(1,i)              =   {N_layerSize(i)};
   NN(3,i)              =   {normrnd(0,sqrt(2/N_layerSize(i)),N_layerSize(i-1),N_layerSize(i))};
   NN(4,i)              =   {zeros(N_layerSize(i),1)};
   v_vect(1,i)          =   {zeros(size(NN{3,i}))};
   v_vect(2,i)          =   {zeros(N_layerSize(i),1)};
end
NN(3,end)               =   {normrnd(0,sqrt(2/N_classes),N_layerSize(end),N_classes)};
NN(4,end)               =   {zeros(N_classes,1)};
v_vect(1,end)           =   {zeros(size(NN{3,end}))};
v_vect(2,end)           =   {zeros(N_classes,1)};

activation_vect         =   cell(1, N_hiddenLayers+1);
delt_vect               =   cell(1, N_hiddenLayers+1);
z_vect                  =   cell(1, N_hiddenLayers+1);
grad                    =   cell(2, N_hiddenLayers+1);

valErrVect              =   zeros(1,N_earlyStop);
errPrev                 =   1;

if(isempty(mu))
    mu                  =   0;
end


%% Data combination
data                    =   [Y,X];

%% Training by SGD
for i = 1:max_epoch
    disp(['Epoch number: ',num2str(i)]);
    % Shuffle data
    data                =   data(randperm(end),:);
    for j = 1:size(data,1)
       % Extract training data and convert as required (required for softmax in the output vector
       x                =   data(j,2:end);
       y                =   data(j,1);
       y_vect           =   zeros(N_classes,1);
       y_vect(y)        =   1;
       
       % Feedforward step
       a                =   x';
       for k = 1:N_hiddenLayers
          W             =   NN{3,k};
          b             =   NN{4,k};
          z             =   W'*a + b;
          a             =   poslin(z);
          activation_vect(1,k)  =   {a};
          z_vect(1,k)   =   {z};
       end
       
       % Output layer
       W                =   NN{3,end};
       b                =   NN{4,end};
       z                =   W'*a + b;
       fout             =   softmax(z);
       activation_vect(1,end)   =   {fout};
       delt             =   softmaxJacobian(z)*softmaxGrad(y_vect,z);
       delt_vect(end)   =   {delt};
       
       %Backpropagation
       for k = N_hiddenLayers:-1:1
          z             =   z_vect{1,k};
          W             =   NN{3,k+1};
          delt          =   ReLuGrad(z)*W*delt;
          delt_vect(k)  =   {delt};
       end
       
       %Final gradients
       grad(1,k)        =   {x' * delt_vect{1}'};
       grad(2,k)        =   delt_vect(1);
       for k = 2:N_hiddenLayers+1
           a            =   activation_vect{k-1};
           delt         =   delt_vect{k};
           grad(1,k)    =   {a*delt'};
           grad(2,k)    =   {delt};
       end
       
       %% Step weights in direction of gradient for each layer
       for k = 1:N_hiddenLayers+1
           W            =   NN{3,k};
           b            =   NN{4,k};
           v_W          =   v_vect{1,k};
           v_b          =   v_vect{2,k};
           grad_W       =   grad{1,k};
           grad_b       =   grad{2,k};
           v_W          =   mu * v_W - eta*grad_W;
           W            =   W + v_W;
           v_b          =   mu * v_b - eta*grad_b;
           b            =   b + v_b;
           
           v_vect(1,k)  =   {v_W};
           v_vect(2,k)  =   {v_b};
           NN(3,k)      =   {W};
           NN(4,k)      =   {b};
       end
       
    end
    
    if(~isempty(data_Val))
        Y_val           =   data_Val(:,1);
        X_val           =   data_Val(:,2:end);
        y_test          =   NN_classify(NN,X_val);
        err             =   sum(Y_val ~= y_test)/length(Y_val);
        valErrVect(1:end-1)         =   valErrVect(2:end);
        valErrVect(end)             =   err >= errPrev;
        errPrev                     =   err;
        if(sum(valErrVect) == N_earlyStop)
            break;
        end
        
    end
end


end

