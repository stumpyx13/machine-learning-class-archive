function f = gaussianRBFKernal(x1, x2)
%gaussianRBFKernal Returns evaluation of the Gaussian RBF function
%   x1 and x2 are column vectors

    global lambda_RBF

    f       =   exp(-lambda_RBF * (x1 - x2)'*(x1 - x2));
end

