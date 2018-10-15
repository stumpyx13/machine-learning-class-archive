function [loss] = logitError_L2Regularizer(w_in)
%logitError_2Normregularizer returns the logistic error with an L2 regularizer

global X Y lambda

% X assumed to have 1 row correspond to 1 observation and Y is assumed to be a column vector
% input w is also a column vector

w_0                 =   w_in(1);
w                   =   w_in(2:end);

N                   =   length(Y);

lossVector          =   1 + exp(-Y .* (X * w + w_0));
lossVector          =   -log(lossVector);

loss                =   -sum(lossVector)/N + lambda * (w'*w);

end

