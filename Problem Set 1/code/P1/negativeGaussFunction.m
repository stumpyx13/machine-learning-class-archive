function [ f ] = negativeGaussFunction( x )
%negativeGaussFunction Calculates the negative Gaussian function for some input x, requires global variable
%specification of the mean vector and covariance matrix

global u covMat

[n,m]                   =   size(covMat);
f                       = - 10^4/(sqrt((2*pi)^n * det(covMat))) * ...
                            exp(-1/2*(x-u)'* (covMat\(x-u)));
end

