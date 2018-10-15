function Y = classify_l1_logreg(X, model)

    %PATH_TO_EXECUTABLES are logreg execs as on website
    PATH_TO_EXECUTABLES     =   ':/usr/local/bin';
    path1 = getenv('PATH');
    if(~contains(getenv('PATH'),'/usr/local/bin'))
        path1 = [path1 PATH_TO_EXECUTABLES];
        setenv('PATH', path1);
    end
    
    %% Run script in the system
    
    mmwrite('in_X',X');
    mmwrite('model_in',model);
    
    system('l1_logreg_classify -q model_in in_X out_Y');
    
    Y = mmread('out_Y');
end

