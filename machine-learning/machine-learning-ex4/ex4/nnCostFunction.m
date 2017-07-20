function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

n = size(X, 2);


K = size(Theta2, 1);
S = 0;

#n - 400
#m - 5000
#size(X) 5000x400
#size(Theta1) 25x401
#size(Theta2) 10x26
#size(y) 5000x1

#size(Theta1_grad);#25x401
#size(Theta2_grad);#10x26

Tgrad1 = 0;
Tgrad2 = 0;



#Backpropagation
delta_sum1=0;
delta_sum2=0;
delta_sum3=0;
#Backpropagation

for i = 1:m
  a1 = [1 X(i,:)];#1x401
  
  yi = y(i);
  yik = zeros(K, 1);
  yik(yi) = 1;
    

  z2 = a1 * Theta1';#1x401*401x25 = 1x25
  a2 = sigmoid(z2);#1x25
    
  Tgrad1 += a2' * a1;# (1x25)' * 1x401 = 25x401
    
  a2 = [1 a2];#1x26
    
  z3 = a2 * Theta2';#1x26*(10x26)' = 1x10
  a3 = sigmoid(z3);#1x10
    
  Tgrad2 += a3' * a2;# (1x10)' * 1x26 = 10x26
    
  #Backpropagation
  delta3 = zeros(K, 1);
  #Backpropagation
    
  for k = 1:K
    
    res = yik(k);#result
    hx = a3(k);#activation of k
    
    S += -1 * res * log(hx) - (1 - res) * log(1 - hx);#1x1
  
    #Backpropagation
    delta3(k) = hx - res;
    #Backpropagation
  end

  #Backpropagation
  
  #delta3 = a3 - yik; # vectorize implementation
  a2_derivative = a2 .* (1 - a2);#1x26 this is equal to sigmoidGradient(z2) + bias
  
  #delta2 = (Theta2' * delta3) .* a2_derivative';# (10x26)' * 10x1 .* (26x1)' = 26x1
 
  
  delta2 = (Theta2' * delta3) .* a2_derivative';# (10x26)' * 10x1 .* (26x1)' = 26x1
  
  #Note that you should skip or remove delta2(0)
  delta2 = delta2(2:end);#25x1
  
  delta_sum2 = delta_sum2 + delta3 * (a2); # 1x10 * 1x26
  delta_sum1 = delta_sum1 + delta2 * (a1); # 26x1 * 1x401
  #size(Theta1) 25x401
  #size(Theta2) 10x26
  #size(Theta1_grad);#25x401
  #size(Theta2_grad);#10x26
  
  #Backpropagation

end

regularization = (lambda / (2*m)) * (getRegularizationFotTheta(Theta1) + getRegularizationFotTheta(Theta2))


J = (1/m) * sum(S) + regularization;

#feedforward
#Theta1_grad = (1/m) * Tgrad1;
#Theta2_grad = (1/m) * Tgrad2;
#feedforward

#Backpropagation
Theta1_without_bias = Theta1;
Theta2_without_bias = Theta2;

#regularizatia should equal to zero for j=0 (j=1 for octave)
Theta1_without_bias(:,1) = 0;
Theta2_without_bias(:,1) = 0;

Theta1_regularization = (lambda/m) * Theta1_without_bias;
Theta2_regularization = (lambda/m) * Theta2_without_bias;

Theta1_grad = (1/m) * delta_sum1 + Theta1_regularization;
Theta2_grad = (1/m) * delta_sum2 + Theta2_regularization;
#Backpropagation

function [regularization] = getRegularizationFotTheta(theta)
  n = size(theta, 2);
  regularVector = zeros(n, n);
  regularVector(1:n+1:n*n) = 1;
  regularVector(1,1) = 0;
  jsum = sum((theta * regularVector) .^ 2);
  ksum = sum(jsum);
  regularization = ksum;
end




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
