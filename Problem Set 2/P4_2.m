%6.867 PSET 2 Problem #4.2
clear;

%% Options
normalizeData                   =   1;
showMisclassifiedDigit          =   1;

%% Specifications

mnist_digits_class1             =   [1,3,5,7,9];
mnist_digits_class2             =   [0,2,4,6,8];

N_train                         =   200;
N_val                           =   150;
N_test                          =   150;

N_vector                        =   784;

baseFile                        =   'hw2_resources/data/mnist_digit_';
extension                       =   '.csv';
%% Initialization

X_train                         =   zeros(N_train*(length(mnist_digits_class1) + length(mnist_digits_class2)),N_vector);
X_val                           =   zeros(N_val*(length(mnist_digits_class1) + length(mnist_digits_class2)),N_vector);
X_test                          =   zeros(N_test*(length(mnist_digits_class1) + length(mnist_digits_class2)),N_vector);

Y_train                         =   zeros(N_train*(length(mnist_digits_class1) + length(mnist_digits_class2)),1);
Y_val                           =   zeros(N_val*(length(mnist_digits_class1) + length(mnist_digits_class2)),1);
Y_test                          =   zeros(N_test*(length(mnist_digits_class1) + length(mnist_digits_class2)),1);

%% Read in data

for i = 1:length(mnist_digits_class1)
   filename                     =   [baseFile, num2str(mnist_digits_class1(i)), extension];
   fileID                       =   fopen(filename);
   A                            =   cell2mat(textscan(fileID, repmat('%f',[1,N_vector])));
   X_train((i-1)*N_train +1:i*N_train,:)    =   A(1:N_train,:);
   X_val((i-1)*N_val +1:i*N_val,:)          =   A((N_train+1):(N_train+N_val),:);
   X_test((i-1)*N_test +1:i*N_test,:)       =   A((N_train+N_val+1):(N_train+N_val+N_test),:);
   
   Y_train((i-1)*N_train +1:i*N_train,1)    =   1;
   Y_val((i-1)*N_val +1:i*N_val,1)          =   1;
   Y_test((i-1)*N_test +1:i*N_test,1)       =   1;
   fclose(fileID);
end

for i = length(mnist_digits_class1)+1:length(mnist_digits_class1)+length(mnist_digits_class2)
   filename                     =   [baseFile, num2str(mnist_digits_class2(i-length(mnist_digits_class1))), extension];
   fileID                       =   fopen(filename);
   A                            =   cell2mat(textscan(fileID, repmat('%f',[1,N_vector])));
   X_train((i-1)*N_train +1:i*N_train,:)    =   A(1:N_train,:);
   X_val((i-1)*N_val +1:i*N_val,:)          =   A((N_train+1):(N_train+N_val),:);
   X_test((i-1)*N_test +1:i*N_test,:)       =   A((N_train+N_val+1):(N_train+N_val+N_test),:);
   
   Y_train((i-1)*N_train +1:i*N_train,1)    =   -1;
   Y_val((i-1)*N_val +1:i*N_val,1)          =   -1;
   Y_test((i-1)*N_test +1:i*N_test,1)       =   -1;
   fclose(fileID);
end

%% Normalize data
if(normalizeData == 1)
    X_train                             =   (2*X_train)./(255) - 1;
    X_val                               =   (2*X_val)./(255) - 1;
    X_test                              =   (2*X_test)./(255) - 1;
end
%% Hyperparameter values
CArray                              =   logspace(-3,3,10);
lambdaArray                         =   logspace(-4,1,10);

%% Train classifier

global X Y kernal lambda_RBF K_gram
X                                   =   X_train;
Y                                   =   Y_train;
kernal                              =   @gaussianRBFKernal;

misRateMatrix                       =   100*ones(length(CArray),length(lambdaArray));
for i = 1:length(CArray)
    for j = 1:length(lambdaArray)
        lambda_RBF                          =   lambdaArray(j);
        K_gram                              =   svm_augmentGram();
        C                                   =   CArray(i);
        [b, a]                              =   softSVM(C);
        
        Y_val_SVM                           =   zeros(size(Y_val));
        for ii = 1:length(Y_val)
            Y_val_SVM(ii)                   =   svmDecision(X_val(ii,:)',a,b);
        end
        N_misclassify_val                   =   sum(Y_val_SVM ~= Y_val);
        misRateMatrix(i,j)                  =   N_misclassify_val/length(Y_val) * 100;
    end
end

[indC, indlambda]                           =   minmat(misRateMatrix);
minError_val                                =   misRateMatrix(indC,indlambda);

lambda_RBF                                  =   lambdaArray(indlambda);
C_opt                                       =   CArray(indC);
[b,a]                                       =   softSVM(C_opt);

%% Test classification accuracy on test data

Y_train_SVM                         =   zeros(length(Y_train),1);
Y_test_SVM                          =   zeros(size(Y_test));

for i = 1:length(Y_test)
    Y_test_SVM(i)                     =   svmDecision(X_test(i,:)',a,b);
end

ind_misclassify_test_SVM          =   find(Y_test_SVM ~= Y_test);
N_misclassify_test_SVM            =   sum( Y_test_SVM ~= Y_test );


disp(['Test SVM misclassification: ', num2str(N_misclassify_test_SVM), ' points (', ...
    num2str(N_misclassify_test_SVM/length(Y_test) * 100), ' %)']);
disp(['\gamma = ',num2str(lambda_RBF),', C = ', num2str(C_opt)]);

%% Plot misclassified digit
if(showMisclassifiedDigit)

    ind_plot                    =   randi([1 length(ind_misclassify_test_SVM)]);
    X_plot                      =   X(ind_misclassify_test_SVM(ind_plot),:);
    titleName                   =   'Misclassified Digit, soft SVM';
   
    
    imageX                          =   reshape(X_plot, [28 28]);
    colormap gray
    imagesc(imageX');
    title(titleName);
    
end
