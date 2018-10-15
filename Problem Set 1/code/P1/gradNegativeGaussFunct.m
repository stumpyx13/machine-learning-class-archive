function [ gradf ] = gradNegativeGaussFunct( x )
%gradNegativeGaussFunct Returns the gradient of the negative Gaussian function. Requires global variable
%specification of the mean vector and covariance matrix
global          u covMat

f               =   negativeGaussFunction(x);

gradf           =   -f * (covMat\(x-u));

end

