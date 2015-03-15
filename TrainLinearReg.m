function w_hat = TrainLinearReg(X,t)
  w_hat = inv(X'* X) * X'* t;
end