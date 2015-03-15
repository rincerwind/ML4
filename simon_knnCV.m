function simon_knnCV(x, t, kVals, Nfold)
  N = size(x,1);

  %% loop over values of K
  Errors = zeros(length(kVals),Nfold);

  order = 1:N;
  sizes = repmat(floor(N/Nfold),1,Nfold);
  sizes(end) = sizes(end) + N - sum(sizes);
  csizes = [0 cumsum(sizes)];

  for kv = 1:length(kVals)
    K = kVals(kv);
    % Loop over folds
    for fold = 1:Nfold
      trainX = x;
      traint = t;
      foldindex = order(csizes(fold)+1:csizes(fold+1));
      trainX(foldindex,:) = [];
      traint(foldindex) = [];
      testX = x(foldindex,:);
      testt = t(foldindex);

      % Do the KNN
      classes = zeros(size(testX,1),1);
      for i = 1:size(testX,1)
        this = testX(i,:);
        dists = sum((trainX - repmat(this,size(trainX,1),1)).^2,2);
        [d I] = sort(dists,'ascend');
        [a,b] = hist(traint(I(1:K)),unique(t));
        pos = find(a==max(a));
          if length(pos)>1
            temp_order = randperm(length(pos));
            pos = pos(temp_order(1));
          end
          classes(i) = b(pos);
      end
      Errors(kv,fold) = sum(classes~=testt);
    end
  end

  %% Plot the results
  figure('Name','Simon KNN');
  s = sum(Errors,2) ./ N;
  plot(kVals,s);
  xlabel('K');
  ylabel('Error');
end