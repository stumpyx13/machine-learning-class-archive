% Problem 2.1 PSET 1 for 6.867 Machine Learning Fall 2017

[x,y]                   =   loadFittingDataP2(0);
hold on;
mArray                  =   [0, 1, 3, 10];

fig                     =   figure(1);
set(fig,'Position',[200, 200, 1600,300]);
for i = 1:length(mArray)
   m                    =   mArray(i);
   beta                 =   calcPolyRegCoef(x,y',m);
   subplot(1,4,i)
   [x,y]                =   loadFittingDataP2(1);
   hold on;
   x_reg                =   linspace(min(x),max(x),1000);
   PHI_reg              =   polyBasis(x_reg,m);
   y_reg                =   PHI_reg*beta;
   plot(x_reg,y_reg)
   title(['Linear regression (M = ', num2str(m),')']);
end