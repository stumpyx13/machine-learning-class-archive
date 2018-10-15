function f = softmaxOutput(a)
    e_vect          =   exp(a);
    f               =   e_vect./sum(e_vect);  
end

