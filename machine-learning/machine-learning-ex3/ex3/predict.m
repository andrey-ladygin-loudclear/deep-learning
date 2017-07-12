function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


#m # 5000
#X = 5000 x 401

alpha1 = X';

z2 = Theta1 * alpha1; # 25x401 * 401x5000 = 25x5000
alpha2 = sigmoid(z2);
alpha2 = [ones(1, m); alpha2];#26x5000

z3 = Theta2 * alpha2;# 10*26 * 26x5000 = 10x5000
alpha3 = sigmoid(z3);



#max(k')'
#max(k, [], 2)


[v p] = max(alpha3', [], 2);



% =========================================================================


end
