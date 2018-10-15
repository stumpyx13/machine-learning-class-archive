function [xx, yy, zz] = plotDecisionBoundary_NN(X, NN, scoreFn)
% X is data matrix (each row is a data point)
% Y is desired output (1 or -1)
% scoreFn is a function of a data point
% values is a list of values to plot

% Plot the decision boundary. For that, we will asign a score to
% each point in the mesh [x_min, m_max]x[y_min, y_max].
    
mins=min(X)-1;
maxes=max(X)+1;
    
h = max((maxes(1)-mins(1))/200., (maxes(2)-mins(2))/200.);
    
[xx, yy] = meshgrid(mins(1):h:maxes(1), mins(2):h:maxes(2));
    
arr=[xx(:),yy(:)];
zz = zeros(length(arr),1);
for i=1:length(arr),
    zz(i) = scoreFn(NN,arr(i,:)); 
end  
zz=reshape(zz,size(xx));

% [C,h]=contour(xx, yy, zz, values);
% set(h,'ShowText','off');
% %Plot the training points
% scatter(X(Y==1,1),X(Y==1,2),40,'bo');
% scatter(X(Y==-1,1),X(Y==-1,2),40,'ro');

    
