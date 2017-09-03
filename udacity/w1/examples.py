from tensorflow.contrib.factorization.examples import mnist
from tensorflow.contrib.keras.python.keras.layers import Dense
from tensorflow.contrib.keras.python.keras.models import Sequential, model_from_json
from tensorflow.contrib.keras.python.keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()

model = Sequential()

#Dense connect all neyrons from this layer to all neyrons from next layer
model.add(Dense(800, input_dim=784, init="normal", activation="relu"))
model.add(Dense(10, init="normal", activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
#optimizer="SGD" stohastic gradient decend, method of learning
#loss="categorical_crossentropy" - error by category
#metrics=["accuracy"] - optimization metric is a accuracy
#metrics=["mae"] - percent of right answers of data set

#mse - mean squared error
#mae - mean absolute error

print(model.summary())



#obuchaem setb | learning
model.fit(X_train, y_train, batch_size=200, np_epoch=100, verbose=1)
#batch_size - number of batch stohastic
#verbose - diagnostic info when model is training66666666666666669
#validation_split=0.2 - 20% for validation set


predictions = model.predict(X_train)

#convert output data to a single number
predictions = np_utils.categorical_probas_to_classes(predictions)


score = model.evaluate(X_test, y_test, verbose=0)
print('Accurate: ', score[1]*100)






##############
model = Sequential()
model.add(Dense(800, input_dim=784, init="normal", activation='relu'))
model.add(Dense(10, init="normal", activation='softmax'))
model_json = model.to_json()
json_file = open('model.json', 'w')
json_file.write(model_json)
json_file.close()

model.save_weights("weighst.h5")

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('weighst.h5')
loaded_model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
