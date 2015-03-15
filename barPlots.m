function barPlots(red, white)  
  figure('Name', 'Red Wine')
  hist(red, unique(red));
  title('Red Wine, Number of Examples vs. Target Values');
  xlabel('Target Values');
  ylabel('Number of Examples');
  
  figure('Name', 'White Wine')
  hist(white, unique(white));
  title('White Wine, Number of Examples vs. Target Values');
  xlabel('Example Numbers');
  ylabel('Target Values');
end