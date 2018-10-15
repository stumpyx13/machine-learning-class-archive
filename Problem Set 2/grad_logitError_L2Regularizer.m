function grad = grad_logitError_L2Regularizer(w_in)
%grad_logitError_L2Regularizer Gradient of the logistic error function with L2 regularizer
% Assumes each observation corresponds to a row of X and Y, X does not include column of ones for bias w_0
% w is a column vector
    global X Y lambda
    
    w_0                 =   w_in(1);
    w                   =   w_in(2:end);
    grad_w              =   0;
    grad_w_0            =   0;
    
    for i = 1:length(Y)
        x               =   X(i,:)';
        grad_w          =   grad_w - Y(i) * x * sigmoid(-Y(i) * (w'*x + w_0) );
        grad_w_0        =   grad_w_0 - Y(i) * sigmoid(-Y(i) * (w'*x + w_0));
    end
    
    grad_w              =   grad_w + lambda*w;
    
    grad                =   [grad_w_0; grad_w];
end

