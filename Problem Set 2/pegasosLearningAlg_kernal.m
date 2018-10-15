function [ alpha ] = pegasosLearningAlg_kernal( lambda, max_iter )
%pegasosLearningAlg_kernal Implements a modified version of Pegasos with kernals based on MIT 6.867 PSET 2 Fall 2017

global X Y kernal

%% Initialization

k               =   0;

[n,m]           =   size(X);
iter            =   0;
alpha           =   zeros(n,1);
X_aug           =    X;
Y_aug           =    Y;
%% Loop

while (iter < max_iter)
    disp(['Epoch #: ',num2str(iter)]);
   for i = 1:n
      k     	=   k + 1;
      eta       =   1/(k*lambda);
      summation     =   0;
      ind           =   find(alpha ~= 0);
      if(isempty(ind))
      else
          for j = 1:length(ind)
             summation  =   summation + alpha(ind(j)) * kernal(X_aug(ind(j),:)',X_aug(i,:)');
          end
      end
      if ((Y_aug(i)*summation) < 1)
         alpha(i,1)   =   (1 - eta*lambda)*alpha(i,1) + eta*Y_aug(i);
      else
         alpha(i,1)   =   (1 - eta*lambda)*alpha(i,1);
      end
   end
%    M            =   [Y_aug,X_aug];
%    M            =   M(randperm(end),:);
%    Y_aug        =   M(:,1);
%    X_aug        =   M(:,2:end);
   iter         =   iter + 1;
  
end
      alpha(alpha < 1e-9) = 0;

end

