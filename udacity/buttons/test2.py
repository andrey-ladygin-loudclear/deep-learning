# License: See LICENSE
# Fit a straight line, of the form y=m*x+b

import tensorflow as tf

'''
Your dataset.
'''
X = [ 0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00] # Features
Y = [ 0.00, 2.00, 4.00, 6.00, 8.00, 10.00, 12.00, 14.00] # Labels

'''
Define free variables to be solved.
'''
m = tf.Variable(0.0) # Parameters
b = tf.Variable(0.0)

'''
Define the error between the data and the model as a tensor (distributed computing).
'''
y_model = m * X + b # Tensorflow knows this is a vector operation
total_error = tf.reduce_sum((Y - y_model)**2) # Sum up every item in the vector

'''
Once cost function is defined, create gradient descent optimizer.
'''
optimizer_operation = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(total_error) # Does one step

'''
Create operator for initialization.
'''
initializer_operation = tf.global_variables_initializer()

'''
All calculations are done in a session.
'''
with tf.Session() as session:

    session.run(initializer_operation) # Call operator

    _EPOCHS = 10000 # number of "sweeps" across data
    for iteration in range(_EPOCHS):
        session.run(optimizer_operation) # Call operator

    slope, intercept = session.run((tf.Variable(12.0), b)) # Call "m" and "b", which are operators
    print((m, b), 'Slope:', slope, 'Intercept:', intercept)