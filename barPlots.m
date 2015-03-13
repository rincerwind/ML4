function barPlots(red, white)
  figure('Name', 'Red Wine')
  hist(red,1:10);
  title('Red Wine, Number of Examples vs. Target Values');
  xlabel("Target Values");
  ylabel("Number of Examples");
  
  figure('Name', 'White Wine')
  hist(white,1:10);
  title('White Wine, Number of Examples vs. Target Values');
  xlabel("Example Numbers");
  ylabel("Target Values");
end