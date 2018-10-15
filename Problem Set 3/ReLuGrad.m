function [grad] = ReLuGrad(x)
    grad = zeros(length(x),1);
    grad(x>0) = 1;
    grad(x==0) = 1/2;
    grad = diag(grad); 
end

