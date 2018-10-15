function [ w ] = pegasosLearningAlg( lambda, max_iter )
%pegasosLearningAlg Implements a modified version of Pegasos based on MIT 6.867 PSET 2 Fall 2017

global X Y

%% Initialization

k               =   0;

[n,m]           =   size(X);
w               =   zeros(m+1,1);
iter            =   0;
w_penalty       =   [0;w(2:end)];
X_aug           =   [ones(n,1), X];
Y_aug           =   Y;
%% Loop

while (iter < max_iter)
   for i = 1:n
      k     	=   k + 1;
      alpha     =   1/(k*lambda);
      if (Y_aug(i)*X_aug(i,:)*w < 1)
         w_k1   =   (1 - alpha*lambda)*w_penalty + alpha*Y_aug(i)*X_aug(i,:)';
         w_k1(1)=   w_k1(1) + w(1);
      else
         w_k1   =   (1 - alpha*lambda)*w_penalty;
         w_k1(1)=   w(1);
      end
      w         =   w_k1;
      w_penalty =   [0;w(2:end)];
   end
   M            =   [Y_aug,X_aug];
   M            =  M(randperm(end),:);
   Y_aug        =   M(:,1);
   X_aug        =   M(:,2:end);
   iter         =   iter + 1;
end

end

