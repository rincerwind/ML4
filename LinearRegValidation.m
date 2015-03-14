function LinearRegValidation(x, t, xTest, xTest_t, maxOrd)
  [x_m, x_n] = size(x);
  [xTest_m, xTest_n] = size(xTest);
  
  % Normalize features
  % This may be a problem, mention it in the report
  
  x(1,:)
  
  x = Normalize(x);
  xTest = Normalize(xTest);
  
  x(1,:)
  train_loss = zeros(1,maxOrd);
  test_loss = zeros(1,maxOrd);
  
  X = x.^0;
  XTest = xTest.^0;
  for i=1:maxOrd
    X = [X, x.^i];
    XTest = [XTest, xTest.^i];
    
    % Compute w on training data
    w = TrainLinearReg(X,t);
    
    % Compute losses
    (train_loss(i)) = LinearRegCost(X, t, w);
    (test_loss(i)) = LinearRegCost(XTest, xTest_t, w);
  end
  
  train_loss
  
  figure('Name', 'Red Validation');
  plot(1:maxOrd, train_loss, "b");
  xlabel("Polynomial Order");
  ylabel("Training Loss");
  
  figure('Name', 'Red Validation');
  plot(1:maxOrd, log10(test_loss), "b");
  xlabel("Polynomial Order");
  ylabel("Log Validation Loss");
end