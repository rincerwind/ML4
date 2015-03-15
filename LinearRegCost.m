function [cost, predict] = LinearRegCost(X, t, w)
  [m, n] = size(X);
  predict = X*w;
  cost = (1/m) * (t - predict)' * (t - predict);
end