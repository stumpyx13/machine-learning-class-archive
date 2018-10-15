function p1 = logist_classify(x, w_in)
% x and w are column vectors
    w_0     =   w_in(1);
    w       =   w_in(2:end);
    
    p1      =   1./(1 + exp(-(w_0 + w'*x)));
end

