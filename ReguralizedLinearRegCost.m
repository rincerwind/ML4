function [cost, predict] = ReguralizedLinearRegCost(X, t, w, lambda)
  [m, n] = size(X);
  cost = LinearRegCost(X,t,w)/2 + (lambda/(2*m))* (w'* w);
  predict = X*w;
end