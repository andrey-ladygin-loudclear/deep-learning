function H = hypotize(X, theta)
  #H = X*theta;
  
  m = length(X);
  H = zeros(m, 1);
  
  for i = 1:m
    x0 = X(i, 1); # 1
    x1 = X(i, 2);
    t1 = theta(1);
    t2 = theta(2);
    t3 = theta(3);
    t4 = theta(4);
    t5 = theta(5);
    
    H(i) = x0*t1 + x1^(1/2)*t2 + x1^2*t3 + x1^3*t4 + x1^5*t5;
  endfor
  
end  

#X =

#    1    1
#    1    2
#    1    4
#    1    7
#    1    8
#    1   11

#ans =  1
#theta =
#
#   0.31420
#   2.29384
#
#ans =  2.2938
#H =

#    3.4578
#    6.4899
#   12.5542
#   21.6506
#   24.6828
#   33.7792