function [train_set, train_set_t, test_set, test_set_t] = splitData(data)
  [m, n] = size(data);
  train_size = round(0.7 * m);
  
  rand_indx = randperm(m);
  train_indx = rand_indx(1:train_size);
  test_indx = rand_indx(train_size + 1:end);
  
  %train_indx = 1:train_size;
  %test_indx = train_size + 1:m;
  
  train_set = data(train_indx,1:end - 1);
  train_set_t = data(train_indx, end);
  
  test_set = data(test_indx,1:end - 1);
  test_set_t = data(test_indx, end);
end