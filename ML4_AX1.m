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
  
  % Init weights
  w_red = zeros(red_n, 1);
  w_white = zeros(white_n, 1);
  
  % Plot the bar-plots
  barPlots(red(:,12), white(:,12));
  
  % Task 1 - Linear Regression -------------------------------------------------------
  % Divide the data sets red wine into training and test sets
  [red_X, red_t, red_Xtest, red_Ttest] = splitData(red);
  
  % Calc rows for red data sets
  red_X_m = size(red_X, 1);
  red_Xtest_m = size(red_Xtest, 1);

  
  % Add biases to the new data sets
  red_X = [ones(red_X_m, 1), red_X];
  red_Xtest = [ones(red_Xtest_m, 1), red_Xtest];

  
  % Fit the linear model
  w_red = TrainLinearReg(red_X, red_t);
  
  % Calc cost and make predictions for test set
  [cost, predictions] = LinearRegCost(red_Xtest, red_Ttest, w_red);
  cost
  
  figure('Name', 'Red Predictions')
  plot(predictions, red_Ttest, ".b");
  title('Red Wine Predictions, Predictions vs. Target Values');
  xlabel("Predictions");
  ylabel("Targets");
  
  % Validate without bias, it is added again inside the function to avoid confusion
  LinearRegValidation(red_X(:,2:end), red_t, red_Xtest(:,2:end), red_Ttest, 8);
  % ---------------------------------------------------------------------------------- 
  
  
  % Task 2 - Reguralized Linear Regression -------------------------------------------
  lambda = -2:0.02:2;
  
  for i=1:length(lambda)
    w = TrainRegularizedLinearReg(red_X, red_t, lambda(i));
    [cost, predictions] = ReguralizedLinearRegCost(red_Xtest, red_Ttest, w, lambda(i));
    err(i) = cost;
  end
  
  figure('Name', 'Reguralized Linear Regression');
  plot(lambda, log10(err), "b-");
  xlabel("Lambda");
  ylabel("Validation Loss");
  
  [min_cost, best_lambda] = CrossValidation([red_X; red_Xtest], [red_t; red_Ttest], 10, 8)
  % ---------------------------------------------------------------------------------- 
end