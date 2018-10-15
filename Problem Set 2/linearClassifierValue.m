function y = linearClassifierValue(X,w_in)
    w_0             =   w_in(1);
    w               =   w_in(2:end);
    
    y               =   w'*X + w_0;
    
end

