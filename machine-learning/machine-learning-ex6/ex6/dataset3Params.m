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

  
#checkerror(X, y, Xval, yval);

C = 1;
sigma = 0.1;
  

function [error] = checkerror(X, y, Xval, yval)
  available = [0.01 0.03 0.1 0.3 1 3 10 30 ];
  available_size = size(available, 2);
  m = size(X, 1);

  x1 = [1 2 1]; x2 = [0 4 -1];
  #model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 

  res = size(available_size^2, 4);


  error_train = zeros(length(available), 1);
  error_val = zeros(length(available), 1);

  index = 1;
  for i = 1:available_size  
    for j = 1:available_size  
      currC = available(i);
      currSigma = available(j);
      
      model = svmTrain(X, y, currC, @(x1, x2) gaussianKernel(x1, x2, currSigma)); 
      
      J_train = svmPredict(model, X);
      J_val = svmPredict(model, Xval);
      predictions = svmPredict(model, Xval);
      
      
      val_error = getError(J_val, yval);
      train_error = getError(J_train, y);
      
      error_train(i, 1) = sum(train_error) / size(train_error, 1);
      error_val(i, 1) = sum(val_error) / size(val_error, 1);
      
      index
      res(index, 1) = mean(double(predictions ~= yval));
      #res(index, 1) = val_error;
      #res(index, 1) = sum(J_val) / size(J_val, 1);
      #res(index, 2) = sum(J_train) / size(J_train, 1);
      res(index, 2) = 0;
      #res(index, 2) = train_error;
      res(index, 3) = currC;
      res(index, 4) = currSigma;
      index = index + 1;
    end 
  end  
    
  res  
  close all;
  plot(available, error_train, available, error_val);
  #plot3(available, available, error_train);
  legend('Train', 'Cross Validation');
  xlabel('lambda');
  ylabel('Error');
  zlabel('Error');
  pause
end    

function [error] = getError(X, y)
  error = sum((X-y).^2);
end

% =========================================================================

end
