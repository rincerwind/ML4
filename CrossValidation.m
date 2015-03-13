function [min_cost, best_lambda] = CrossValidation(X, t, K)
  [m, n] = size(X);
  %X(:,2:end) = Normalize(X(:,2:end));
 
  fold_size = floor(m/K);
  
  % shuffle the data to increase randomness
  %new_indecies = randperm(m);
  %x = x(new_indecies,:);
  %t = t(new_indecies,:);
  
  lambda = -2:0.02:2;
  
  %X = [ones(m,1), x];
  for i=1:length(lambda)
    fold_start = 1;
    fold_end = fold_start + fold_size - 1;
    
    for k=2:K
      testX = X(fold_start:fold_end,:);
      testT = t(fold_start:fold_end,:);
      
      trainX = X;
      trainX(fold_start:fold_end,:) = [];
      trainT = t;
      trainT(fold_start:fold_end,:) = [];
      
      w = TrainRegularizedLinearReg(trainX, trainT, lambda(i));
      [cost, predictions] = ReguralizedLinearRegCost(testX, testT, w, lambda(i));
     
      cv_loss(k-1, i) =  cost;
      
      fold_start = fold_end + 1;
      if( fold_start + fold_size - 1 > m || k == K )
        fold_end = m;
      else
        fold_end = fold_start + fold_size - 1;
      end
    end % end of folding loop
  end % end of order loop
  
  mean_losses = mean(cv_loss,1);
  [min_loss, min_indx] = min(mean_losses);
  min_cost = min_loss;
  best_lambda = lambda(min_indx);
  
  figure('Name', '10-Fold Cross Validation');
  plot(lambda, log10(mean_losses));
  xlabel("Lambda");
  ylabel("Validation Loss");
end