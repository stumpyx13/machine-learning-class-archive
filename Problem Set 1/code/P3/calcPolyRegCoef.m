function [ beta ] = calcPolyRegCoef( x, y, m )
%calcPolyRegCoef Calculates the regression coefficients for a polynomial regression of order m
    n                   =   length(x);
    PHI                 =   polyBasis(x,m);
    beta                =   (PHI'*PHI)\(PHI'*y);

end

