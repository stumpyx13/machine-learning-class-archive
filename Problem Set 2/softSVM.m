function [b,a_opt] = softSVM( C )
%softSVM soft margin SVM, calculates weight vector, bias, and Lagrange multipliers
% X and Y must be specified as global variables
% X is a matrix whose rows correspond to observations and Y is a column vector
% kernal function must be specified as global

global X Y kernal K_gram

%% Optimization parameters

ub              =   C * ones(size(Y));
lb              =   zeros(size(Y));

obj             =   @dualSoftSVM_Objective;

a0              =   ones(size(Y));

A               =   [];
b               =   [];
Aeq             =   Y';
beq             =   0;

%% Calculate multipliers

%a_opt           =   fmincon(obj,a0,A,b,Aeq,beq,lb,ub);
a_opt           =   quadprog(K_gram,-ones(size(Y)),A,b,Aeq,beq,lb,ub,a0);

a_opt(a_opt < 1e-9) = 0;

%% Calculate weight vector
% w               =   0;
% 
% for i = 1:length(Y)
%     w           =   w + a_opt(i) * Y(i) * X(i,:)';
% end

b               =   0;
for i = 1:length(a_opt)
    if(a_opt(i) ~= 0)
        b           =   b + Y(i);
        secondSum   =   0;
        for j = 1:length(a_opt)
            if(a_opt(j) ~= 0)
                secondSum   =   secondSum + a_opt(j)*Y(j)*kernal(X(i,:)',X(j,:)');
            end
        end
        b           =   b - secondSum;
    end
end

b                   =   b / sum(abs(Y(a_opt ~= 0)));

end

