clear ; close all; clc

data = [
  1 2;
  2 8;
  4 20;
  7 28;
  8 29;
  11 32;
];


data = [
  1 1;
  2 2.2;
  3 3.5;
  4 4;
  5 4.2;
  6 20.4;
  7 21.6
  8 23.9
  9 38
  10 42.2
];

X = data(:, 1);
y = data(:, 2);
m = length(y);


X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(5, 1);
X
#J = computeCost(X, y, theta);
iterations = 2000;
alpha = 0.00000000003;

[theta, J_history] = gradientDescent(X, y, theta, alpha, iterations);

#figure;
#plot(1:numel(J_history), J_history, 'b', 'LineWidth', 2);

figure;
plot(X(:,2), y, 'rx', 'MarkerSize', 10);
ylabel('Price');
xlabel('Size');


res = hypotize(X, theta);

#data
theta
#X(:,2)
#res

hold on;
plot(X(:,2), res, '-')
hold off;
