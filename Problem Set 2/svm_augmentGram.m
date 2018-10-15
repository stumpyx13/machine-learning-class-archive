function [ K ] = svm_augmentGram()

global X kernal Y

[n,m]           =   size(X);

for i = 1:n
    for j = 1:n
        K(i,j)  =   Y(i) * Y(j) * kernal(X(i,:)',X(j,:)');
    end
end


end

