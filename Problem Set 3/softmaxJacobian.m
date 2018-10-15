function J = softmaxJacobian(x)
    x           =   reshape(x,[length(x),1]);
    S           =   softmax(x);
    J           =   -S*S';
    J           =   J + diag(S);
end

