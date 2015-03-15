function LinearRegValidation(x, t, xTest, xTest_t, maxOrd)
  [x_m, x_n] = size(x);
  [xTest_m, xTest_n] = size(xTest);
  
  train_loss = zeros(1,maxOrd);
  test_loss = zeros(1,maxOrd);
  
  x = Normalize(x);
  xTest = Normalize(xTest);
  
  X = ones(x_m,1);
  XTest = ones(xTest_m,1);
  
  for i=1:maxOrd
    X = [X, x.^i];
    XTest = [XTest, xTest.^i];
    
    % Compute w on training data
    w = TrainLinearReg(X,t);
    
    % Compute losses
    (train_loss(i)) = LinearRegCost(X, t, w);
    (test_loss(i)) = LinearRegCost(XTest, xTest_t, w);
  end
  
  figure('Name', 'Red Validation');
  plot(1:maxOrd, train_loss, "b-o");
  xlabel("Polynomial Order");
  ylabel("Training Loss");
  
  figure('Name', 'Red Validation');
  plot(1:maxOrd, log10(test_loss), "b-o");
  xlabel("Polynomial Order");
  ylabel("Log Validation Loss");
end