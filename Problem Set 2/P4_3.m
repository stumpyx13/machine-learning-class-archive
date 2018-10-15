%6.867 PSET 2 Problem #4.2
clear;

%% Options
normalizeData                   =   1;

%% Specifications

mnist_digits_class1             =   [4];
mnist_digits_class2             =   [9];



N_vector                        =   784;

baseFile                        =   'hw2_resources/data/mnist_digit_';
extension                       =   '.csv';

global X Y max_epochs lambda_Pegasos C K_gram

C                               =   1;

%% Loop parameters
N_train_array                   =   [200; 300; 400; 500]/2;
time_QP_SVM                     =   zeros(size(N_train_array));
time_pegasos                    =   zeros(size(N_train_array));
N_misclassifyRate_QP            =   zeros(size(N_train_array));

epochArray                      =   [10^1, 10^2, 500];
lambdaArray                     =   [1e-3,1e-2,1e-1];
legendEntries                   =   [];
hold on

for ii = 1:length(epochArray)
    max_epochs                  =   epochArray(ii);
    for iii = [1,3]
        lambda_Pegasos          =   lambdaArray(iii);
        for j = 1:length(N_train_array)
            %% Initialization
            N_train                   =   N_train_array(j);
            X                         =   zeros(N_train*(length(mnist_digits_class1) + length(mnist_digits_class2)),N_vector);

            Y                         =   zeros(N_train*(length(mnist_digits_class1) + length(mnist_digits_class2)),1);

            %% Read in data

            for i = 1:length(mnist_digits_class1)
               filename                     =   [baseFile, num2str(mnist_digits_class1(i)), extension];
               fileID                       =   fopen(filename);
               A                            =   cell2mat(textscan(fileID, repmat('%f',[1,N_vector])));
               X((i-1)*N_train +1:i*N_train,:)    =   A(1:N_train,:);

               Y((i-1)*N_train +1:i*N_train,1)    =   1;
               fclose(fileID);
            end

            for i = length(mnist_digits_class1)+1:length(mnist_digits_class1)+length(mnist_digits_class2)
               filename                     =   [baseFile, num2str(mnist_digits_class2(i-length(mnist_digits_class1))), extension];
               fileID                       =   fopen(filename);
               A                            =   cell2mat(textscan(fileID, repmat('%f',[1,N_vector])));
               X((i-1)*N_train +1:i*N_train,:)    =   A(1:N_train,:);


               Y((i-1)*N_train +1:i*N_train,1)    =   -1;
               fclose(fileID);
            end

            %% Normalize data
            if(normalizeData == 1)
                X                             =   (2*X)./(255) - 1;
            end
            
            %% Find time to train each SVM
            
            if(ii == 1 && iii == 1)
                tic
                Y_train                     =   zeros(size(Y));
                K_gram      =   svm_augmentGram();
                [b,a]                       =   softSVM(C);
                time_QP_SVM(j)               =  toc;
                for jj = 1:length(Y_train)
                   Y_train(jj)              =   svmDecision(X(jj,:)',a,b); 
                end
                N_misclassifyRate_QP(j)     =   sum(Y_train ~= Y)/length(Y);
            end
            tic
            alpha                           =   pegasosLearningAlg_kernal(lambda_Pegasos,max_epochs);            
            time_pegasos(j)                 =   toc;
        end
        if(ii == 1 && iii ==1)
           plot(N_train_array,time_QP_SVM);
        end
        plot(N_train_array,time_pegasos);
                  
    end
end

