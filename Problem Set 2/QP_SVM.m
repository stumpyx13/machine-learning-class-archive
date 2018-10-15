function  [] =  QP_SVM()

global C K_gram

K_gram      =   svm_augmentGram();

[b,a]       =   softSVM(C);
    
end

