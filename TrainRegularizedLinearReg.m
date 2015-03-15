function w_hat = TrainRegularizedLinearReg(X, t, lambda)
  [m, n] = size(X);
  identity = eye(n);
  
  w_hat = inv(X'* X + m*lambda*identity)*X'*t;
end