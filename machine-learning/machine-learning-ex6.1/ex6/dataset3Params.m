function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.3;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
available = [0.01 0.03 0.1 0.3 1 3 10 30 ];
available_size = size(available, 2);
m = size(X, 1);

x1 = [1 2 1]; x2 = [0 4 -1];
model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 

res = size(available_size^2, 3);

index = 1;
for i = 1:available_size  
  for j = 1:available_size  
    currC = available(i);
    currSigma = available(j);
    
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
    J_val = svmPredict(model, Xval);
    error = getError(J_val, yval);
    
    index
    res(index, 1) = error;
    res(index, 2) = currC;
    res(index, 3) = currSigma;
    index = index + 1;
  end 
end  
  
#res
min_err = min(res(:,1));
m_err = size(res, 1);
for i = 1:m_err
  if res(i,1) == min_err
    res(i,:)
  end
end

C = 30;
sigma = 0.3;

#C = 0.01;
#sigma = 0.01;
pause
  
#for i = 1:m
#  X_train = X(1:i, :);
 # y_train = y(1:i);
#  X_val = Xval;
#  y_val = yval;
#  
#  #[theta] = trainLinearReg(X_train, y_train, lambda);
#  model = svmTrain(X_train, y_train, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
#  
#  J_train = svmPredict(model, X_train);
#  J_val = svmPredict(model, X_val);
#  
#  #[J_train] = linearRegCostFunction(X_train, y_train, theta, 0);
#  #[J_val] = linearRegCostFunction(X_val, y_val, theta, 0);
#  
#  error_train(i) = getError(J_train, y_train);
#  error_val(i) = getError(J_val, y_val);
#  
#end#

#hold off;
#fprintf('plotData !!!!!!!!!!! ...\n')
#plot(1:m, error_train, 1:m, error_val);
#pause;
#fprintf('plotData2 222 ...\n')
#plotData(error_train, error_val);
#pause;#

#C
#sigma

function [error] = getError(X, y)
  error = sum((X-y).^2);
end

% =========================================================================

end
