function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

n = length(theta);
regularVector = zeros(n);
regularVector(1:n+1:n*n) = 1;
regularVector(1,1) = 0;

#theta
#regularVector*theta

#lambda = 10;

data = X * theta;
h = sigmoid(data);

#LTHETA = theta
#UUU = (regularVector*theta)
#UUU2 = (regularVector*theta) .^ 2


J = (1/m) * (-1*y'*log(h) - (1 - y)' * log(1 - h)) + (lambda / (2*m)) * sum((regularVector*theta) .^ 2);
#J = (1/m) * (-1*y'*log(h) - (1 - y)' * log(1 - h));


grad = (1/m) * X' * (h - y) + (lambda / m) * (regularVector*theta);
#grad = (1/m) * X' * (h - y);



% =============================================================

end
