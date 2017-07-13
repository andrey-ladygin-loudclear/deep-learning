clear ; close all; clc

data = [
  0 1;
  1 0.98;
  2 0.97;
  3 0.96;
  4 0.95;
  5 0.94;
  6 0.93;
  7 0.92;
  8 0.91;
  9 0.9;
  10 0.88;
  11 0.85;
  12 0.8;
  13 0.75;
  14 0.72;
  15 0.68;
  16 0.4;
  17 0.3;
  18 0.2;
  19 0.1;
  20 0;
]

X = data(:, 1); 
y = data(:, 2);
m = length(y);

plotData(X, y);

X = [ones(m, 1), X];

theta = (X'*X)' * X'*y

hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-')
hold off % don't overlay any more plots on this figure






function [J, grad] = costFunction(theta, X, y)
  m = length(y); % number of training examples
  data = X * theta;
  h = sigmoid(data);

  J = (1/m) * (-1*y'*log(h) - (1 - y)' * log(1 - h));

  grad = (1/m) * X' * (h - y);
end
