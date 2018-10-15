% Problem 3.1 for PSET 1 6.867 Machine Learning Fall 2017
[x,y]               =   loadFittingDataP2(1);
mArray              =   [1, 2, 4, 7, 8];

lambda              =  3;
MSE                 =   zeros(1, length(mArray));

for i = 1:length(mArray)
    m = mArray(i);
    PHI                 =   cosBasis(x,m);

    theta               =   (eye(size(PHI'*PHI)) * lambda + PHI'*PHI)\(PHI'*y');

    hold on

    x_reg               =   linspace(min(x),max(x),1000);
    PHI_reg             =   cosBasis(x_reg,m);
    y_reg               =   PHI_reg*theta;
    y_guess             =   PHI*theta;
    
    MSE(i)              =   norm(y_guess - y',2)^2/length(x);

    plot(x_reg,y_reg)

end
set(gca,'FontSize',16);
xlabel('x','FontSize',16);
ylabel('y','FontSize',16);
title(['\lambda = ',num2str(lambda)])
legend('Data points','M = 1','M = 2','M = 4','M = 7', 'M = 8')
%legend('Data points','Maximum likelihood regression')