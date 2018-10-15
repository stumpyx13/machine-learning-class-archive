% Problem 1.1 for 6.867 PSET 1, September 2017

%% Load data

global A b u covMat

[gaussMean, gaussCov, quadBowlA, quadBowlb] = loadParametersP1();

A                                           =   quadBowlA;
b                                           =   quadBowlb;
u                                           =   gaussMean;
covMat                                      =   gaussCov;

%% Options
investigateNegGauss                         =   0;
investigateQuadBowl                         =   1;

%% Constants and initialization
d                                           =   [1,0.5,0.1,0.005,0.001,0.0005,0.0001];
err_G                                       =   ones(1,length(d))*10;
err_qb                                      =   ones(1,length(d))*10;

for j = 1:length(d)
    %% Choose random points
    [n_G,m_G]                                   =   size(gaussMean);
    [n_qb,m_qb]                                 =   size(quadBowlb);
    N_points                                    =   100;
    randPoints_G                                =   zeros(n_G,N_points);
    randPoints_qb                               =   zeros(n_qb,N_points);
    for i = 1:n_G
       randPoints_G(i,:)                        =   10*(rand(1,N_points) - 0.5);
    end
    for i = 1:n_qb
       randPoints_qb(i,:)                       =   10*(rand(1,N_points) - 0.5);
    end

    %% Calculate analytical and numerical gradients

    gradReal_G                                  =   zeros(n_G,N_points);
    gradCDA_G                                   =   zeros(n_G,N_points);
    gradReal_qb                                 =   zeros(n_qb,N_points);
    gradCDA_qb                                  =   zeros(n_qb,N_points);

    for i = 1:N_points
        x                                       =   randPoints_G(:,i);
        gradReal_G(:,i)                         =   gradNegativeGaussFunct(x);
        for k = 1:n_G
            x_above                             =   x;
            x_above(k)                          =   x(k)+d(j);
            x_below                             =   x;
            x_below(k)                          =   x(k)-d(j);
            gradCDA_G(k,i)                      =   (negativeGaussFunction(x_above) ...
                                                     - negativeGaussFunction(x_below))/(2*d(j));
        end
        y                                       =   randPoints_qb(:,i);
        gradReal_qb(:,i)                        =   quadbowlGrad(y);
        for k = 1:n_qb                                         
            y_above                                 =   y;
            y_above(k)                              =   y(k)+d(j);
            y_below                                 =   y;
            y_below(k)                              =   y(k)-d(j); 
            gradCDA_qb(k,i)                         =   (quadraticBowlFunct(y_above) ...
                                                    -quadraticBowlFunct(y_below))/(2*d(j));
        end
    end
    
    %% Calculate errors
    errSum_G                                    =   0;
    errSum_qb                                   =   0;
    
    for i = 1:N_points
       errSum_G                                 =   errSum_G + norm(gradReal_G(:,i) - gradCDA_G(:,i),2);
       errSum_qb                                =   errSum_qb + norm(gradReal_qb(:,i) - gradCDA_qb(:,i),2);
    end
    errMean_G                                   =   errSum_G/N_points;
    errMean_qb                                  =   errSum_qb/N_points;
    
    err_G(j)                                    =   errMean_G;
    err_qb(j)                                   =   errMean_qb;
end
%% Run Gradient descent

if(investigateNegGauss == 1)
    figure(1);
    loglog(d,err_G,'b-*');
    xlabel('\delta','FontSize',16);
    ylabel('Mean squared error','FontSize',16);
    set(gca,'FontSize',16);
    set(gca,'Xdir','reverse');
    title('Central difference error for negative Gaussian function')
    grid on;
end


if(investigateQuadBowl == 1)
    figure(2);
    loglog(d,err_qb,'b-*');
    xlabel('\delta','FontSize',16);
    ylabel('Mean squared error','FontSize',16);
    set(gca,'FontSize',16);
    %set(gca,'Xdir','reverse');
    title('Central difference error for quadratic bowl function')
    grid on;
    
end



