function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

#h = X*theta

J = sum((X*theta - y) .^ 2) / (2*m);
#error: computeCost: operator *: nonconformant arguments (op1 is 1x2, op2 is 97x2)

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


#fprintf('\n PARAMS !!!!!!!!!!!!!!!!\n')


#fprintf('\n N=%f', m);
#fprintf(X);
#fprintf(y);
#fprintf(theta);

#fprintf('\n PARAMS !!!!!!!!!!!!!!!!\n')

#J = (1 / 2*m) * sum(theta'*X - y);


% =========================================================================

end
