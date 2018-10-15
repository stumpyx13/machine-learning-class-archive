% Problem 2.2 for PSET 1 for 6.867 Machine Learning Fall 2017

[x,y]               =   loadFittingDataP2(0);
n                   =   length(y);
d                   =   0.001;
mArray              =   [0, 1, 3, 10];
SSE                 =   zeros(1,length(mArray));
err_grad            =   zeros(1,length(mArray));

global X Y
Y                   =   y';

for i = 1:length(mArray)
    m                   =   mArray(i);
    beta                =   calcPolyRegCoef(x,y',m);
    PHI                 =   polyBasis(x,m);
    y_reg               =   PHI*beta;
    X                   =   PHI;
    SSE(i)              =   sumSquaresError_linearReg(beta);
    grad_anal           =   grad_leastSquaresErr_linearReg(beta);
    grad_CDA            =   grad_centralDiffApprox(@sumSquaresError_linearReg,beta,d);
    err_grad(i)         =   norm(grad_anal - grad_CDA,2);
end

