function [ PHI ] = sinBasis( x, m )
%Applies a polynomial basis of order m to vector x
    n                   =   length(x);
    PHI                 =   ones(n,m+1);
    for i = 1:n
          PHI(i,1)      =   x(i);
       for j = 1:m
          PHI(i,j+1)    =   sin(0.4* pi * j * x(i)); 
       end
    end


end

