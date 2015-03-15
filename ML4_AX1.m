function ML4_AX1()
  % Read the CSV files
  red = dlmread("winequality-red.csv", ";");
  white = dlmread("winequality-white.csv", ";");
  
  % Remove first row as it holds 0s because of the labels in the CSV file
  red = red(2:end,:);
  white = white(2:end,:);
  
  % Cals sizes
  [red_m, red_n] = size(red);
  [white_m, white_n] = size(white);
  
  % Normalize all features, except the target values
  %red(:,1:end-1) = Normalize( red(:,1:end-1) );
  %white(:,1:end-1) = Normalize( white(:,1:end-1) );
  
  % Plot the bar-plots
  barPlots(red(:,end), white(:,end));
  
  % Task 1 - Linear Regression -------------------------------------------------------
  % Divide the data sets red wine into training and test sets
  [red_X, red_t, red_Xtest, red_Ttest] = splitData(red);
  
  % Calc rows for red data sets
  red_X_m = size(red_X, 1);
  red_Xtest_m = size(red_Xtest, 1);
  
  % Fit the linear model
  w_red = TrainLinearReg([ones(red_X_m, 1), red_X], red_t);
  
  % Calc cost and make predictions for test set
  [cost, predictions] = LinearRegCost([ones(red_Xtest_m, 1), red_Xtest], red_Ttest, w_red);
  cost
  
  figure('Name', 'Red Predictions')
  plot(predictions, red_Ttest, ".b");
  title('Red Wine Predictions, Predictions vs. Target Values');
  xlabel("Predictions");
  ylabel("Targets");
  
  % Validate without bias, it is added again inside the function to avoid confusion
  LinearRegValidation(red_X, red_t, red_Xtest, red_Ttest, 7);
  % ---------------------------------------------------------------------------------- 
  
  
  % Task 2 - Reguralized Linear Regression -------------------------------------------
  lambda = [-0.5, -0.1, -0.05, -0.01, 0, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 25];
  loss = ones(length(lambda),1);
  
  for i=1:length(lambda)
    w = TrainRegularizedLinearReg([ones(red_X_m, 1), red_X], red_t, lambda(i));
    [loss(i), predictions] = ReguralizedLinearRegCost([ones(red_Xtest_m, 1), red_Xtest], red_Ttest, w, lambda(i));
  end
  
  figure('Name', 'Reguralized Linear Regression');
  plot(lambda, loss, "b-o");
  title('70/30 Validation, Reguralized Linear Regression');
  xlabel("Lambda");
  ylabel("Validation Loss");
  
  % Validate without bias, it is added again inside the function to avoid confusion
  [min_cost_reguralized, best_lambda] = CrossValidation(red_X, red_t, red_Xtest, red_Ttest, 10, lambda)
  % ---------------------------------------------------------------------------------- 
  
  % Task 3 - K-Nearest Neighbours ----------------------------------------------------
  
  % Do a 10-Fold KNN
  %knnCV([red_X; red_Xtest], [red_t; red_Ttest], [1, 5, 10, 30, 50, 100], 10);
  %simon_knnCV([red_X; red_Xtest], [red_t; red_Ttest], [1, 5, 10, 30, 50, 100], 10);
  % ---------------------------------------------------------------------------------- 
end