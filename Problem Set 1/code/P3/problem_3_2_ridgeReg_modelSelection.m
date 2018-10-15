% Problem 3.2 for PSET 1 6.867 Machine Learning Fall 2017
%% Options
modelSelection      =   0;
testModel           =   1;
trainingData        =   2; %1 = A, 2 = B

%% Chosen model
lambda_model        =   8;
m_model             =   5;

%% Load data
[xA,yA]             =   regressAData();
[xB,yB]             =   regressBData();
[xV,yV]             =   validateData();

if(modelSelection)
    mArray              =   [1, 2, 3, 4, 7];

    lambda              =  8;
    MSE                 =   zeros(1, length(mArray));
    figure(1)
    switch(trainingData)
        case(1)
            x           =   xA;
            y           =   yA;
        case(2)
            x           =   xB;
            y           =   yB;
    end
    scatter(x,y)
    
    figure(2)
    scatter(xV,yV)
    for i = 1:length(mArray)
        m = mArray(i);
        %% Train data
        
        PHI                 =   polyBasis(x,m);

        theta               =   (eye(size(PHI'*PHI)) * lambda + PHI'*PHI)\(PHI'*y');
        x_train             =   linspace(min(x),max(x),1000);
        PHI_train           =   polyBasis(x_train,m);
        y_train             =   PHI_train*theta;
        figure(1)
        hold on
        plot(x_train,y_train)
      
        
        

        %% Plot against validation data
        x_reg               =   linspace(min(xV),max(xV),1000);
        PHI_reg             =   polyBasis(x_reg,m);
        y_reg               =   PHI_reg*theta;
        PHI_V               =   polyBasis(xV,m);
        y_guess             =   PHI_V*theta;

        MSE(i)              =   norm(y_guess - yV',2)^2/length(xV);
        figure(2)
        hold on
        plot(x_reg,y_reg)

    end
    figure(2)
    set(gca,'FontSize',16);
    xlabel('x','FontSize',16);
    ylabel('y','FontSize',16);
    title(['Validation data, \lambda = ',num2str(lambda)])
    legend('Data points','M = 1','M = 2','M = 3','M = 4','M = 7')
    
    figure(1)
    set(gca,'FontSize',16);
    xlabel('x','FontSize',16);
    ylabel('y','FontSize',16);
    title(['Training data, \lambda = ',num2str(lambda)])
    legend('Data points','M = 1','M = 2','M = 3','M = 4','M = 7')
end

if(testModel)
    switch(trainingData)
        case(1)
            x           =   xA;
            y           =   yA;
            xT          =   xB;
            yT          =   yB;
        case(2)
            x           =   xB;
            y           =   yB;
            xT          =   xA;
            yT          =   yA;
    end
    % Determine weights
    PHI                 =   polyBasis(x,m_model);

    theta               =   (eye(size(PHI'*PHI)) * lambda_model + PHI'*PHI)\(PHI'*y');
    
    % Test against test data
    PHI_T               =   polyBasis(xT,m_model);
    y_guess_test        =   PHI_T * theta;
    MSE_T               =   norm(y_guess_test - yT',2)^2/length(xT);
    
    % Plot against test data
    x_reg_T             =   linspace(min(xT),max(xT),1000);
    PHI_reg_T           =   polyBasis(x_reg_T,m_model);
    y_reg_T             =   PHI_reg_T*theta;
    
    figure(3)
    hold on
    set(gca,'FontSize',16);
    xlabel('x','FontSize',16);
    ylabel('y','FontSize',16);
    scatter(xT,yT);
    plot(x_reg_T,y_reg_T)
    legend('Test data','Chosen model')
    title(['Chosen model M = ',num2str(m_model), ', \lambda = ',num2str(lambda_model)]);
end
%legend('Data points','Maximum likelihood regression')