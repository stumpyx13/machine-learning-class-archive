function [ err ] = leastSquaresError_linearReg( beta )
%leastSquaresError Returns the least squares error for a linear regression
%   Detailed explanation goes here
global X Y
[~,m]           =   size(beta);

for     i = 1:m
    err(i)             =   (Y - X*beta(:,i))' * (Y - X*beta(:,i));
end

end

