function L = softmaxLoss(a,y)
    L = -sum(y.*log(softmax(a)));
end

