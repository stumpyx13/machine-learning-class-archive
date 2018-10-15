function [ theta ] = ridgeRegress( X, y, lambda )
%ridgeRegress ridge regression of data with regularizer lambda
    
    pseudoX         =   X'*X;
    theta           =   (eye(size(pseudoX)) * lambda + pseudoX)\(X'*y);


end

