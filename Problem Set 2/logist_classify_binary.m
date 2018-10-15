function p1 = logist_classify_binary(x, w_in)
% x and w are column vectors
    w_0     =   w_in(1);
    w       =   w_in(2:end);
    
    p1      =   1./(1 + exp(-(w_0 + w'*x)));
    
    
    if(p1 > 0.5)
        p1  =   1;
    else
        p1  =   -1;
    end
    
end

