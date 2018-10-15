function Y = NN_classify(NN,X)
    
    N_hiddenLayers                  =   size(NN,2) - 1;
    N_data                          =   size(X,1);
    Y                               =   zeros(N_data,1);
    for i = 1:N_data
       % Feedforward
       a                            =   X(i,:)';
       for k = 1:N_hiddenLayers
          W             =   NN{3,k};
          b             =   NN{4,k};
          z             =   W'*a + b;
          a             =   poslin(z);
       end
        
       %Output layer
       W                =   NN{3,end};
       b                =   NN{4,end};
       z                =   W'*a + b;
       fout             =   softmax(z);
       [~,ind]          =   max(fout);
       
       %Assign
       Y(i)             =   ind;
    end
end

