function [cost, predict] = LinearRegCost(X, t, w)
  [m, n] = size(X);
  cost = (1/(2*m)) * (t - X*w)' * (t - X*w);
  predict = X*w;
end