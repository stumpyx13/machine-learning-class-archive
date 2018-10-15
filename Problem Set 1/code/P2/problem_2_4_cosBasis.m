% Problem 2.4 for PSET 1 6.867 Machine Learning Fall 2017
clear; clc;
[x,y]               =   loadFittingDataP2(1);
m                   =   8;

PHI                 =   cosBasis(x,m);

theta               =   (PHI'*PHI)\(PHI'*y')

hold on

x_reg               =   linspace(min(x),max(x),1000);
PHI_reg             =   cosBasis(x_reg,m);
y_reg               =   PHI_reg*theta;

plot(x_reg,y_reg)
set(gca,'FontSize',16);
xlabel('x','FontSize',16);
ylabel('y','FontSize',16);
legend('Data points','Maximum likelihood regression')