function norm_X = Normalize(x)
  [m,n] = size(x);
  
  %norm_X = (x - mean(x))/std(x);
  %norm_X = bsxfun(@minus, x, mean(x));
  %norm_X = bsxfun(@rdivide, bsxfun(@minus, x, mean(x)), std(x));
  
  means = mean(x);
  stds = std(x);

  for i=1:n
    x(:,i) = (x(:,i) - means(i)) / stds(i);
  end
  
  norm_X = x;
end