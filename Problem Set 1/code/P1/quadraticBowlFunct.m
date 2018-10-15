function [ f ] = quadraticBowlFunct( x )
%quadraticBowlFunct Returns the value of the quadratic bowl function for some input x. Requires global
%variable specification of matrix A and vector b.
global A b

f           =   1/2 * x'*A*x - x'*b;

end

