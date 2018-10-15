% 6.867 PSET 2 Problem 2.2

%% Set global variables
clear;
global X Y kernal K_gram lambda_RBF

kernal          =   @gaussianRBFKernal;
lambda_RBF      =   1;

%% Load data
dataSetN                    =   4;
dataFilePath                =   ['hw2_resources/data/data',num2str(dataSetN),'_train.csv'];
fileID                      =   fopen(dataFilePath);
data                        =   textscan(fileID,'%f %f %f');
data                        =   cell2mat(data);
fclose(fileID);

X                           =   data(:,1:2);
Y                           =   data(:,3);

dataFileValidate            =   ['hw2_resources/data/data',num2str(dataSetN),'_validate.csv'];
fileID                      =   fopen(dataFileValidate);
data_val                    =   textscan(fileID,'%f %f %f');
data_val                    =   cell2mat(data_val);
fclose(fileID);

X_val                       =   data_val(:,1:2);
Y_val                       =   data_val(:,3);

%% SVM parameters
K_gram          =   svm_augmentGram();


C               =   1;

[b,a_opt]     =   softSVM(C);

%% Calculate misclassification for training and validation sets

Y_training                  =   zeros(size(Y));
Y_validation                =   zeros(size(Y_val));

for i = 1:length(Y_training)
   Y_training(i)            =   svmDecision(X(i,:)',a_opt,b); 
end

for i = 1:length(Y_validation)
   Y_validation(i)          =   svmDecision(X_val(i,:)',a_opt,b); 
end

Y_training(Y_training > 0)            = 1;
Y_training(Y_training <= 0)           = -1;

Y_validation(Y_validation > 0)        = 1;
Y_validation(Y_validation <= 0)       = -1;

similarity_train                        =   Y == Y_training;
similarity_validation                   =   Y_val == Y_validation;

N_misclassified_train                   =   length(Y) - sum(similarity_train);
N_misclassified_validation              =   length(Y_val) - sum(similarity_validation);

%% Make plot
hold on
% scatter(X(Y==1,1),X(Y==1,2),'bo');
% scatter(X(Y==-1,1),X(Y==-1,2),'ro');

for i = 1:length(a_opt)
   if(a_opt(i) ~= 0)
      if(Y(i) == 1)
        scatter(X(i,1),X(i,2),'b*')
      else
        scatter(X(i,1),X(i,2),'r*')
      end
   end
end
a               =   a_opt;
% x1              =   linspace(min(X(:,1)),max(X(:,1)),100);
% 
% for i = 1:length(x1)
%     x2_test     =   linspace(min(X(:,2)*2),max(X(:,2)*2),1000);
%     x_test      =   [x1(i)*ones(1000,1), x2_test'];
%     for j = 1:length(x2_test)
%         y_test(j)  =   svmDecision(x_test(j,:)', a, b);
%     end
%     [~,ind]         =   min(abs(y_test));
%     x2(i)           =   x2_test(ind);
%     [~,ind2]        =  min(abs(y_test-1));
%     x2_marg1(i)     =   x2_test(ind2);
%     [~,ind3]        =   min(abs(y_test+1));
%     x2_marg2(i)     =   x2_test(ind3);
% end
% 
% ind1                =   find(x2 > 2*min(X(:,2)) & x2 < 2*max(X(:,2)));
% x1_marg1            =   x1;
% x1_marg2            =   x1;
% x1                  =   x1(ind1);
% x2                  =   x2(ind1);
% 
% ind1                =   find(x2_marg1 > 2*min(X(:,2)) & x2_marg1 < 2*max(X(:,2)));
% x1_marg1            =   x1_marg1(ind1);
% x2_marg1            =   x2_marg1(ind1);
% 
% ind2                =   find(x2_marg2 > 2*min(X(:,2)) & x2_marg2 < 2*max(X(:,2)));
% x2_marg2            =   x2_marg2(ind2);
% x1_marg2            =   x1_marg2(ind2);


%x2              =   (-x1*w(1) - b)/w(2);
h = figure(1);
%plot(x1,x2,'k',x1_marg1,x2_marg1,'k--',x1_marg2,x2_marg2,'k--')
values              =   [-1,0,1];
[xx,yy,zz] = plotDecisionBoundary(X,Y,a_opt,b,@svmDecision,values,['data ',num2str(dataSetN),' training, C = ', num2str(C),...
    '; Gaussian RBF Kernel (\lambda_{RBF} = ', ...
    num2str(lambda_RBF),')']);
set(gca,'FontSize',16);
xlabel('x_1','FontSize',18);
ylabel('x_2','FontSize',18);
%legend('+1 points','-1 points','+1 support vector', '-1 support vector')


hh = figure(2);
hold on
colormap cool
[con,hhh]=contour(xx, yy, zz, values);
set(hhh,'ShowText','off');
%Plot the training points
scatter(X_val(Y_val==1,1),X_val(Y_val==1,2),40,'bo');
scatter(X_val(Y_val==-1,1),X_val(Y_val==-1,2),40,'ro');
set(gca,'FontSize',16);
xlabel('x_1','FontSize',18);
ylabel('x_2','FontSize',18);
%legend('+1 points','-1 points')
title(['data ',num2str(dataSetN),' validation, C = ', num2str(C),'; Gaussian RBF Kernel (\lambda_{RBF} = ', ...
    num2str(lambda_RBF),')']);

disp(['Misclassified points in training: ',num2str(N_misclassified_train),'(',...
    num2str(N_misclassified_train/length(Y) * 100),'%)'])
disp(['Misclassified points in validation: ',num2str(N_misclassified_validation),'(',...
    num2str(N_misclassified_validation/length(Y_val) * 100),'%)'])
