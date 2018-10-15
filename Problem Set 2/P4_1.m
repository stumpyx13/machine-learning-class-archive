%6.867 PSET 2 Problem #4
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

%% Train classifier

global X Y kernal lambda_RBF lambda K_gram

kernal                              =   @gaussianRBFKernal;
lambda_RBF                          =   .001;
lambda                              =   0.01;
X                                   =   X_train;
Y                                   =   Y_train;

w0                                  =   ones(N_vector + 1,1) * 0.01;

% soft SVM - traditional
K_gram                              =   svm_augmentGram();
C                                   =   10;
[b, a]                              =   softSVM(C);

% LR - ridge
options                             =   optimoptions(@fminunc,'MaxFunctionEvaluations',length(w0)*1000);
w                                   =   fminunc(@logitError_L2Regularizer,w0, options);
                                            

%% Test classification accuracy on training and test data

Y_train_LR                          =   zeros(length(Y_train),1);
Y_train_SVM                         =   zeros(length(Y_train),1);

Y_test_LR                           =   zeros(size(Y_test));
Y_test_SVM                          =   zeros(size(Y_test));

for i = 1:length(Y_train)
    Y_train_LR(i)                      =   logist_classify_binary(X_train(i,:)',w);
    Y_train_SVM(i)                     =   svmDecision(X_train(i,:)',a,b);
end

for i = 1:length(Y_test)
    Y_test_LR(i)                      =   logist_classify_binary(X_test(i,:)',w);
    Y_test_SVM(i)                     =   svmDecision(X_test(i,:)',a,b);
end

ind_misclassify_train_LR           =   find(Y_train_LR ~= Y_train);
ind_misclassify_train_SVM          =   find(Y_train_SVM ~= Y_train);

N_misclassify_train_LR             =   sum( Y_train_LR ~= Y_train );
N_misclassify_train_SVM            =   sum( Y_train_SVM ~= Y_train );

ind_misclassify_test_LR           =   find(Y_test_LR ~= Y_test);
ind_misclassify_test_SVM          =   find(Y_test_SVM ~= Y_test);

N_misclassify_test_LR             =   sum( Y_test_LR ~= Y_test );
N_misclassify_test_SVM            =   sum( Y_test_SVM ~= Y_test );

disp(['Training LR misclassification: ', num2str(N_misclassify_train_LR), ' points (', ...
    num2str(N_misclassify_train_LR/length(Y_train) * 100), ' %)']);
disp(['Training SVM misclassification: ', num2str(N_misclassify_train_SVM), ' points (', ...
    num2str(N_misclassify_train_SVM/length(Y_train) * 100), ' %)']);

disp(['Test LR misclassification: ', num2str(N_misclassify_test_LR), ' points (', ...
    num2str(N_misclassify_test_LR/length(Y_test) * 100), ' %)']);
disp(['Test SVM misclassification: ', num2str(N_misclassify_test_SVM), ' points (', ...
    num2str(N_misclassify_test_SVM/length(Y_test) * 100), ' %)']);

%% Plot misclassified digit
if(showMisclassifiedDigit)
    %rng(0,'twister');
    chooser                         =   randi([0 1]);
    if(chooser == 0)
        ind_plot                    =   randi([1 length(ind_misclassify_test_LR)]);
        X_plot                      =   X(ind_misclassify_test_LR(ind_plot),:);
        titleName                   =   'Misclassified Digit, Logistic Regression';
    else
        ind_plot                    =   randi([1 length(ind_misclassify_test_SVM)]);
        X_plot                      =   X(ind_misclassify_test_SVM(ind_plot),:);
        titleName                   =   'Misclassified Digit, soft SVM';
    end
    
    imageX                          =   reshape(X_plot, [28 28]);
    colormap gray
    imagesc(imageX');
    title(titleName);
    
end
