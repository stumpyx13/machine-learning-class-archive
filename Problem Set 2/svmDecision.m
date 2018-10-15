function y = svmDecision(x, a, b)
%svmDecision returns y(x) (7.13 in Bishop)
global X Y kernal

y           =   0;

SV_ind      =   find(a~=0);

for i = 1:length(SV_ind)
    y       =  y + a(SV_ind(i)) * Y(SV_ind(i)) * kernal(x,X(SV_ind(i),:)');
end
y           =   y + b;

if(y <= 0)
    y = -1;
else
    y = 1;
end

end

