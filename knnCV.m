function knnCV(X, t, kVals, N) % N is the fold
  [m, n] = size(X);
  
  fold_size = floor(m/N);
  unique_t = unique(t);
  
  fold_start = ones(1,N);
  fold_end = ones(1,N);
  fold_end = fold_start + fold_size - 1;
  
  % pre-compute fold sizes
  for fold=2:N
    fold_start(fold) = fold_end(fold-1) + 1;
    if( fold_start(fold) + fold_size - 1 > m || fold == N )
       fold_end(fold) = m;
    else
       fold_end(fold) = fold_start(fold) + fold_size - 1;
    end
  end
  
  % Loop over values of K for KNN
  for k=1:length(kVals)
    K = kVals(k);
    
    % Loop over the N folds
    for fold=1:N
      fStart = fold_start(fold);
      fEnd = fold_end(fold);
      
      testX = X(fStart:fEnd,:);
      testT = t(fStart:fEnd,:);
      
      trainX = X;
      trainX(fStart:fEnd,:) = [];
      trainT = t;
      trainT(fStart:fEnd,:) = [];
    
      [train_m, train_n] = size(trainX);
      [test_m, test_n] = size(testX);
       
      % KNN for all test cases
      c_predict = zeros(test_m,1);
      for i=1:test_m
          % replicate testX(i,:) in train_m row, so we can efficiently 
          % find the distances from the current testX to all trainX points
          rep_testX = repmat(testX(i,:), train_m, 1);
          
          % sum row-wise as 1 row is 1 point
          dists = sum((trainX - rep_testX) .^ 2, 2);
          
          % sorts dists and returns the sorted array, 
          % as well as previous indecies of the current elements
          [sDists, inds] = sort(dists,'ascend');
          [vals, bins] = hist(trainT(inds(1:K)), unique_t);
          
          % max can return the index of the max element, 
          % but there can be many equal max elements
          (max_val) = max(vals);
          max_pos = find(vals == max_val);
          
          % there can be more than one class with a max
          % in this case choose randomly
          if length(max_pos)>1
            max_pos = max_pos(1);
            %rand_max = randperm(length(max_pos));
            %max_pos = max_pos(rand_max(1));
          end
          c_predict(i) = bins(max_pos);
      end
      
      % perform 0/1 loss (Accuracy)
      err(k, fold) = sum(c_predict~=testT)/length(testT);
    end % end of fold loop
  end % end of K loop
  
  figure('Name', '10-Fold KNN');
  plot(kVals, mean(err,2), 'b-o');
  title('10-Fold Cross Validation, KNN');
  xlabel('K');
  ylabel('Error');
end % end of function