function LinearRegValidation(x, t, xTest, xTest_t, maxOrd)
  [x_m, x_n] = size(x);
  [xTest_m, xTest_n] = size(xTest);
  
  % Normalize features
  % This may be a problem, mention it in the report
  x(:,2:end) = Normalize(x(:,2:end));
  xTest(:,2:end) = Normalize(xTest(:,2:end));
  
  for i=1:maxOrd
    X = [ones(x_m,1)];
    XTest = [ones(xTest_m,1)];
    
    for j = 1:i
      X = [X, x.^j];
      XTest = [XTest, xTest.^j];
    end
    
    % Compute w on training data
    w = TrainLinearReg(X,t);
    
    % Compute losses
    [ train_loss(i), predictX ] = LinearRegCost(X, t, w);
    [ test_loss(i), predictXtest ] = LinearRegCost(XTest, xTest_t, w);
  end
  
  figure('Name', 'Red Validation');
  plot(1:maxOrd, train_loss, "b");
  xlabel("Polynomial Order");
  ylabel("Training Loss");
  
  figure('Name', 'Red Validation');
  plot(1:maxOrd, log10(test_loss), "b");
  xlabel("Polynomial Order");
  ylabel("Log Validation Loss");
end