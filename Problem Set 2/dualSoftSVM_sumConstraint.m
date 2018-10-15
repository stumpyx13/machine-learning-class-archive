function [fineq,f] = dualSoftSVM_sumConstraint(a)
%dualSoftSVM_sumConstraint sum constraint for the dual form of the soft SVM (equation 7.34 in Bishop)
global Y
fineq       =   [];
f           =   a .* Y;
f           =   sum(f);

end

