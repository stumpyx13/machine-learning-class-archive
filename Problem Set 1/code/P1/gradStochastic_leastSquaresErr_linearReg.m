function [ grad ] = gradStochastic_leastSquaresErr_linearReg( beta )
%grad_leastSquaresErr_linearReg Summary of this function goes here
%   Note: X is the data, n x m, beta is m x 1
    global  X Y calls loops usedInd
    
    [n,m]                   =   size(X);
     if (rem(calls+1,n) == 0 || calls == 0)
        loops = 0;
        usedInd     =   [0];
     end
    loops = loops + 1;
    ind_st              =   randi([1,n],1);
    while(ismember(ind_st,usedInd))
        ind_st              =   randi([1,n],1);
    end
    x_vect              =   X(loops,:);
    y_vect              =   Y(loops);
    deltaGrad           =   -2 * (y_vect - x_vect*beta)*x_vect';
    grad                =    deltaGrad;

end

