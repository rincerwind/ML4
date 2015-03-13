function w_hat = TrainLinearReg(X,t)
  w_hat = pinv(X'*X)*X'*t;
end