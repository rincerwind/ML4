function ML4_AX1()
  %% Initial data processing
  
  % Read the CSV files
  red = dlmread('winequality-red.csv', ';',1,0);
  white = dlmread('winequality-white.csv', ';',1,0);
  
  % Plot the bar-plots
  % by obtaining unique bins and counting their occurence frequencies
  figure('Name', 'Red Wine');
  hist(red(:,end), unique(red(:,end))); 
  title('Red Wine, Frequency vs. Target Values');
  xlabel('Target Values');
  ylabel('Frequency');
  
  figure('Name', 'White Wine');
  hist(white(:,end), unique(white(:,end)));
  title('White Wine, Frequency vs. Target Values');
  xlabel('Target Values');
  ylabel('Frequency');
  
  % (WORD)
  % The majority of the data points is concentrated in 2 classes, in the 
  % Red Wine data set, 
  
  %% Task 1 - Linear Regression
  
  % Shuffle the Red Wine data set and split it into 70%/30% data sets
  [m, n] = size(red);
  train_size = round(0.7 * m);
  
  rand_indx = randperm(m);
  %train_indx = rand_indx(1:train_size);
  %test_indx = rand_indx(train_size + 1:end);
  
  train_indx = 1:train_size;
  test_indx = train_size + 1:m;
  
  red_X = red(train_indx,1:end - 1);
  red_t = red(train_indx, end);
  
  red_Xtest = red(test_indx,1:end - 1);
  red_Ttest = red(test_indx, end);
  
  red_X_m = size(red_X, 1);
  red_Xtest_m = size(red_Xtest, 1);
  
  % Fit a linear model to the training set
  X = [ones(red_X_m, 1), red_X];
  t = red_t;
  w_red = (X' * X) \ (X' * t);
  
  % Calc loss and make predictions for test set
  XTest = [ones(red_Xtest_m, 1), red_Xtest];
  TTest = red_Ttest;
  
  m = size(XTest,1);
  predict = XTest * w_red;
  linearReg_loss = ( 1/(2*m) ) * ((TTest - predict)' * (TTest - predict))
  
  figure('Name', 'Red Predictions')
  plot(predict, red_Ttest, '.b');
  title('Red Wine Predictions, Predictions vs. Target Values');
  xlabel('Predictions');
  ylabel('Targets');
  
  % Benchmark, using higher order polynomial models
  % -----------------------------------------------------------------------
  x = red_X;
  t = red_t;
  xTest = red_Xtest;
  xTest_t = red_Ttest;
  maxOrd = 7;
  
  x_m = size(x,1);
  xTest_m = size(xTest,1);
  
  train_loss = zeros(1,maxOrd);
  test_loss = zeros(1,maxOrd);
  
  % Normalize the data as the values get pretty high
  x = Normalize(x);
  xTest = Normalize(xTest);
  
  % Add biases
  X = ones(x_m,1);
  XTest = ones(xTest_m,1);
  
  for i=1:maxOrd
    X = [X, x.^i];
    XTest = [XTest, xTest.^i];
    
    % Compute w on training data
    w = (X' * X) \ (X' * t);
    
    % Compute losses
    predict = X * w;
    train_loss(i) = ( 1/(2*size(X,1)) ) * ((t - predict)' * (t - predict));

    predict = XTest * w;
    test_loss(i) = ( 1/(2*size(X,1)) ) * ... 
        ((xTest_t - predict)' * (xTest_t - predict));
  end
  
  figure('Name', 'Red Validation');
  plot(1:maxOrd, train_loss, 'b-o');
  xlabel('Polynomial Order');
  ylabel('Training Loss');
  
  figure('Name', 'Red Validation');
  plot(1:maxOrd, log10(test_loss), 'b-o');
  xlabel('Polynomial Order');
  ylabel('Log Validation Loss');
  % -----------------------------------------------------------------------
  
  
  %% Task 2 - Reguralized Linear Regression -------------------------------
  lambdas = [0, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10];
  loss = ones(length(lambdas),1);
  
  X = [ones(red_X_m, 1), red_X];
  t = red_t;
  
  xTest = [ones(red_Xtest_m, 1), red_Xtest];
  xTest_t = red_Ttest;
  
  % Try out models with different values of lambda
  for i=1:length(lambdas)
    l = lambdas(i);
    [m, n] = size(X);
    identity = eye(n);
    w = ((X' * X) + m * l * identity) \ (X' * t);
    
    predict = xTest * w;
    loss(i) = ( 1/(2*size(xTest,1)) ) * ... 
        ((xTest_t - predict)' * (xTest_t - predict));
  end
  
  figure('Name', 'Reguralized Linear Regression');
  plot(lambdas, loss, 'b-o');
  title('70%/30% Validation, Reguralized Linear Regression');
  xlabel('Lambda');
  ylabel('Loss');
  
  % 10-Fold Cross Validation
  % -----------------------------------------------------------------------
  x = red_X;
  t = red_t;
  
  testX = red_Xtest;
  testT = red_Ttest;
  
  m = size(x, 1);
  m_test = size(testX, 1);
  
  X = [ones(m,1), x];
  testX = [ones(m_test,1), testX];
  
  K = 10;
  
  % Init loss tables
  train_loss = zeros(length(lambdas),K);
  cv_loss = zeros(length(lambdas),K);
  test_loss = zeros(length(lambdas),K);
 
  fold_size = floor(m/K);
  
  fold_start = ones(1,K);
  fold_end = ones(1,K);
  fold_end(1) = fold_start(1) + fold_size - 1;
  
  % Pre-compute fold sizes
  for fold=2:K
    fold_start(fold) = fold_end(fold-1) + 1;
    if( fold_start(fold) + fold_size - 1 > m || fold == K )
       fold_end(fold) = m;
    else
       fold_end(fold) = fold_start(fold) + fold_size - 1;
    end
  end
  
  for i=1:length(lambdas)
    l = lambdas(i);
    for k=1:K
      fStart = fold_start(k);
      fEnd = fold_end(k);
      
      testX_fold = X(fStart:fEnd,:);
      testT_fold = t(fStart:fEnd,:);
      
      trainX_fold = X;
      trainX_fold(fStart:fEnd,:) = [];
      trainT_fold = t;
      trainT_fold(fStart:fEnd,:) = [];
      
      identity = eye(size(X,2));
      w = ((trainX_fold' * trainX_fold) + size(trainX_fold,1) * ... 
          l * identity) \ (trainX_fold' * trainT_fold);
      
      predict = trainX_fold * w;
      train_loss(i, k) = (1/(2*size(trainX_fold,1))) * ...
          ((trainT_fold - predict)' * (trainT_fold - predict));
      
      predict = testX_fold * w;
      cv_loss(i, k) = (1/(2*size(testX_fold,1))) * ...
          ((testT_fold - predict)' * (testT_fold - predict));
      
      predict = testX * w;
      test_loss(i, k) = (1/(2*size(testX,1))) * ...
          ((testT - predict)' * (testT - predict));
    end % end of folding loop
  end % end of order loop
  
  mean_Train = mean(train_loss,2);
  mean_CV = mean(cv_loss,2);
  mean_Test = mean(test_loss,2);
  
  size(mean_Train)
  
  [min_loss, min_indx] = min(mean_Test);
  redLinearReg_loss = min_loss
  best_lambda = lambdas(min_indx)
  
  figure('Name', '10-Fold Cross Validation');
  hold on;
  plot(lambdas, mean_Train, 'b-o');
  plot(lambdas, mean_CV, 'r-o');
  plot(lambdas, mean_Test, 'k-o');
  legend('Mean Train Loss', 'Mean CV Loss', 'Mean Test Loss');
  title('10-Fold Cross Validation, Reguralized Linear Regression');
  xlabel('Lambda');
  ylabel('Validation Loss');
  hold off;
  % -----------------------------------------------------------------------
  
  %% Task 3 - K-Nearest Neighbours ----------------------------------------
  
  % Do a 10-Fold KNN
  knnCV([red_X; red_Xtest], [red_t; red_Ttest], [1, 3, 7, 11, 19], 10);
  % -----------------------------------------------------------------------
end