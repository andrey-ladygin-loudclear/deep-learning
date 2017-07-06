function J = computeCost(X, y, theta)

m = length(y);

H = hypotize(X, theta);
  
a = 100;
  
J = sum((H - y) .^ 2 + a*sum(theta)) / (2*m);

end