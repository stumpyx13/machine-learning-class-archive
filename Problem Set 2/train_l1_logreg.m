function [model] = train_l1_logreg(X, Y, lambda)
%train_l1_logreg trains l1_logreg function from Stanford

    %PATH_TO_EXECUTABLES are logreg execs as on website
    PATH_TO_EXECUTABLES     =   ':/usr/local/bin';
    path1 = getenv('PATH');
    if(~contains(getenv('PATH'),'/usr/local/bin'))
        path1 = [path1 PATH_TO_EXECUTABLES];
        setenv('PATH', path1);
    end
    
    mmwrite('ex_X',X);
    mmwrite('ex_Y',Y);
    
    system(['l1_logreg_train -s ex_X ex_Y ', num2str(lambda), ' model_iono']);
    
    model           =   mmread('model_iono');

end

