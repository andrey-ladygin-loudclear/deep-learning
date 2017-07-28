function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = cost(X, theta);
diff = (h - y) .^ 2;

J = (1/(2*m)) * (sum(diff)) + getRegularization(theta, lambda, m);

grad = (1/m) * X' * (h - y) + getGradRegularization(theta, lambda, m);


function [res] = cost(x, theta)
  res = x * theta;
end  

function [regularization] = getRegularization(theta, lambda, m)
  regular_theta = theta;
  regular_theta(1) = 0;
  regularization = (lambda/(2*m)) * sum(regular_theta .^ 2);
end  

function [regularization] = getGradRegularization(theta, lambda, m)
  regular_theta = theta;
  regular_theta(1) = 0;
  regularization = (lambda/m) * sum(regular_theta);
end  



% =========================================================================

grad = grad(:);

end
