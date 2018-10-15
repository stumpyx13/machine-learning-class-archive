function [ grad ] = grad_leastSquaresErr_linearReg( beta )
%grad_leastSquaresErr_linearReg Summary of this function goes here
%   Note: X is the data, n x m, beta is m x 1
global X Y
    
    [n,m]                   =   size(X);
    grad                    =   0;
    for i = 1:n
        x_vect              =   X(i,:);
        y_vect              =   Y(i);
        deltaGrad           =   -2 * (y_vect - x_vect*beta)*x_vect';
        grad                =    grad + deltaGrad;
    end

end

