import nn
import numpy as np

Network = nn.NN()
X = np.array([
    [1,2,3,4,5,6,7],
    [1,2,3,4,5,6,7]
])
Y = np.array([
    [3,6,9,12,15,18,21],
    [1,4,9,16,25,36,49]
])
Network.set_X(X)
Network.set_Y(Y)

#L1 = nn.Layer(X.shape[0], Y.shape[0], act=np.tanh, act_derivative=nn.tanh_deriviate)
L1 = nn.Layer(X.shape[0], 4, act=lambda x:x, act_derivative=lambda x:x)
L2 = nn.Layer(4, Y.shape[0], act=lambda x:x)
#L2 = nn.Layer(4, Y.shape[0], act=lambda x:x)
Network.add_layer(L1)
Network.add_layer(L2)
#Network.add_layer(L2)
Network.nn_model(num_iterations=20000,print_cost=True,learning_rate=0.000008)
predictions = Network.predict(np.array([
    [15],
    [8]
]))
#predictions = predict(parameters, )
#print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
print(predictions)