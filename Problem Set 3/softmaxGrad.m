function grad = softmaxGrad(y, a)
    grad = -y./(softmax(a));
end

