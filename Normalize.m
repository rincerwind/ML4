function norm_X = Normalize(x)
  norm_X = bsxfun(@rdivide, bsxfun(@minus, x, mean(x)), std(x));
end