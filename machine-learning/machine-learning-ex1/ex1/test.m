clear ; close all; clc

fprintf('Loading data ...\n');

#http://td.chem.msu.ru/uploads/files/courses/special/expmethods/statexp/LabLecture02.pdf

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(data);
#min = min(data(:,:));
#max = max(data(:,:));
min = min(X);
max = max(X);
average = max / 2;

disp(X(10, :))
X = (X - average) / (max - min);
disp(X(10, :))