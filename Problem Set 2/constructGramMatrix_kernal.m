function [ K ] = constructGramMatrix_kernal()

global X kernal

[n,m]           =   size(X);

for i = 1:n
    for j = 1:n
        K(i,j)  =   kernal(X(i,:)',X(j,:)');
    end
end


end

