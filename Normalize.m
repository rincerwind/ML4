function norm_X = Normalize(x)
  [m,n] = size(x);
  means = mean(x);
  stds = std(x);

  for i=1:n
    x(:,i) = (x(:,i) - means(i)) / stds(i);
  end
  
  norm_X = x;
end