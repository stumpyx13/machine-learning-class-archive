function [ grad_CDA ] = grad_centralDiffApprox( f, x, d )
%grad_sumSquaresErr_centralDiffApprox Returns the central difference approximation for a function at a specified point

[n,m]                   =   size(x);
% if(n > m)
%     x                   =   x';
%     [n,m]                   =   size(x);
% end

grad_CDA                =   zeros(n,m);

for i = 1:m
    x_i                 =   x(:,i);
    for j = 1:n
        x_CDA_above     =   x_i;
        x_CDA_above(j)  =   x_i(j) + d;
        x_CDA_below     =   x_i;
        x_CDA_below(j)  =   x_i(j) - d;
        grad_CDA(j,i)   =   (f(x_CDA_above) - f(x_CDA_below))/(2*d);
    end
    
end
end

