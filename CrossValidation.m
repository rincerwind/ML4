function [min_cost, best_lambda] = CrossValidation(x, t, testX, testT, K, lambda)
  [m, n] = size(x);
  [m_test, n_test] = size(testX);
 
  fold_size = floor(m/K);
  
  fold_start = ones(1,K);
  fold_end = ones(1,K);
  fold_end = fold_start + fold_size - 1;
  
  % pre-compute fold sizes
  for fold=2:K
    fold_start(fold) = fold_end(fold-1) + 1;
    if( fold_start(fold) + fold_size - 1 > m || fold == K )
       fold_end(fold) = m;
    else
       fold_end(fold) = fold_start(fold) + fold_size - 1;
    end
  end
  
  X = [ones(m,1), x];
  testX = [ones(m_test,1), testX];
  for i=1:length(lambda)
    for k=1:K
      fStart = fold_start(k);
      fEnd = fold_end(k);
      
      testX_fold = X(fStart:fEnd,:);
      testT_fold = t(fStart:fEnd,:);
      
      trainX_fold = X;
      trainX_fold(fStart:fEnd,:) = [];
      trainT_fold = t;
      trainT_fold(fStart:fEnd,:) = [];
      
      w = TrainRegularizedLinearReg(trainX_fold, trainT_fold, lambda(i));
      (train_loss(i, k)) = ReguralizedLinearRegCost(trainX_fold, trainT_fold, w, lambda(i));
      (cv_loss(i, k)) = ReguralizedLinearRegCost(testX_fold, testT_fold, w, lambda(i));
      (test_loss(i, k)) = ReguralizedLinearRegCost(testX, testT, w, lambda(i));
    end % end of folding loop
  end % end of order loop
  
  mean_Train = mean(train_loss,2);
  mean_CV = mean(cv_loss,2);
  mean_Test = mean(test_loss,2);
  
  [min_loss, min_indx] = min(mean_Test);
  min_cost = min_loss;
  best_lambda = lambda(min_indx);
  
  figure('Name', '10-Fold Cross Validation');
  hold on;
  plot(lambda, mean_Train, "b-o");
  plot(lambda, mean_CV, "r-o");
  plot(lambda, mean_Test, "k-o");
  title('10-Fold Cross Validation, Reguralized Linear Regression');
  xlabel("Lambda");
  ylabel("Validation Loss");
  hold off;
end