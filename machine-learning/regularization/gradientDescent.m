function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

m = length(y);
J_history = zeros(num_iters, 1);

newX = [X, X(:,2) .^ 2];
#newX = X


for iter = 1:num_iters

  H = hypotize(X, theta);

  newTheta = zeros(m, 1);
  
  #diff = (H - y)'
  #(H - y)' * X
  #((H - y)' * X)' 
 
  #theta = theta - (alpha/m) * ((H - y)' * X)';
  theta = theta - (alpha/m) * ((H - y)' * newX)';
  
  
  #diff = (alpha/m) * ((H - y)' * X)';
  #theta(1) = theta(1) - diff(1);
  #theta(2) = theta(2) - diff(2);
  
  
  #s = 0;
  #diff = H - y;
  #diff
  #X
  #C = (H - y)' * X
  #for i = 1:m
  #  sum += diff(i) * X(i)
  #end  

  J_history(iter) = computeCost(X, y, theta);

end

end
