% 6.867 PSET 1 Problem 2.1

%% Set global variables
clear;
global X Y kernal K_gram

kernal          =   @linearKernal;

%% Load data

X               =   [2,2; 2,3; 0,-1; -3,-2];
Y               =   [1; 1; -1; -1];

%% SVM parameters
K_gram          =   svm_augmentGram();


C               =   1e32;

[w,b,a_opt]     =   softSVM(C);

%% Make plot
hold on
scatter(X(1:2,1),X(1:2,2),'bo');
scatter(X(3:4,1),X(3:4,2),'ro');

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
x1              =   linspace(min(X(:,1)),max(X(:,1)),1000);

for i = 1:length(x1)
    x2_test     =   linspace(min(X(:,2)*2),max(X(:,2)*2),1000);
    x_test      =   [x1(i)*ones(1000,1), x2_test'];
    for j = 1:length(x2_test)
        y_test(j)  =   svmDecision(x_test(j,:)', a, b);
    end
    [~,ind]         =   min(abs(y_test));
    x2(i)           =   x2_test(ind);
    [~,ind2]        =  min(abs(y_test-1));
    x2_marg1(i)     =   x2_test(ind2);
    [~,ind3]        =   min(abs(y_test+1));
    x2_marg2(i)     =   x2_test(ind3);
end

%x2              =   (-x1*w(1) - b)/w(2);
plot(x1,x2,'k',x1,x2_marg1,'k--',x1,x2_marg2,'k--')
set(gca,'FontSize',16);
xlabel('x_1','FontSize',18);
ylabel('x_2','FontSize',18);
legend('+1 points','-1 points','+1 support vector', '-1 support vector','Decision boundary')