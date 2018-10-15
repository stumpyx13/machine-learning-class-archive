function [loss] = dualSoftSVM_Objective(a)
%dualSoftSVM_Objective Dual formulation of soft SVM objective function, formualted for convex optimization
%(min -objective instead of max objective)

%   a is a column vector
%   X is a matrix where each row is an observation
%   Y is a column vector

global X Y kernal

Loss_term1              =   sum(a);
Loss_term2              =   0;

for i = 1:length(Y)
    for j = 1:length(Y)
        Loss_term2      =   Loss_term2 + a(i)*a(j)*Y(i)*Y(j)*kernal(X(i,:)',X(j,:)');
    end
end

loss                    =   -Loss_term1 + 0.5 * Loss_term2;

end

