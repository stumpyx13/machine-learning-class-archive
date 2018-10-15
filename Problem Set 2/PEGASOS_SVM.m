function [] = PEGASOS_SVM()

    global lambda_Pegasos max_epochs
    [alpha] = pegasosLearningAlg_kernal(lambda_Pegasos, max_epochs);
end

