function y = kernelClassifierValue(x_in,alpha)
%kernelClassifierValue Summary of this function goes here
    global kernal X Y
    
    y       =   0;
    ind     =   find(alpha ~= 0);
    for i = 1:length(ind)
        %kernel(X(ind(i),:)',x_in)
        y   =   y + alpha(ind(i)) * kernal(X(ind(i),:)',x_in);
    end
end

