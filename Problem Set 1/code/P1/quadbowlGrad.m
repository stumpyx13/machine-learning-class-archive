function [ gradf ] = quadbowlGrad( x )
%quadBowlGrad Returns the gradient of the quadratic bowl function at some vector x. Requires global variable
%specification of matrix A and vector b.
global A b

gradf               =   A*x-b;

end

