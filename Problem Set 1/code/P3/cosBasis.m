function [ PHI ] = cosBasis( x, m )
%Applies a polynomial basis of order m to vector x
    n                   =   length(x);
    PHI                 =   ones(n,m+1);
    for i = 1:n
       for j = 0:m
          PHI(i,j+1)      =   cos(pi * j * x(i)); 
       end
    end


end

