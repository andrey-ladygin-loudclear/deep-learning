from tensorflow.contrib.learn import LinearRegressor, pandas_input_fn, DNNRegressor, Experiment
from tensorflow.python.feature_column.feature_column import categorical_column_with_hash_bucket, numeric_column, \
    categorical_column_with_vocabulary_list, embedding_column, indicator_column

make = categorical_column_with_hash_bucket('make', 100)
horsepower = numeric_column('horsepower', shape=[])
cylinders = categorical_column_with_vocabulary_list('num-of-cylinders', ['two', 'three', 'four', 'six', 'eight'])


###############
regressor = DNNRegressor(feature_columns=[embedding_column(make, 10),
                                          horsepower,
                                          indicator_column(cylinders, 3)],
                         hidden_units=[50, 30, 10])
################
regressor = LinearRegressor(feature_columns=[make, horsepower, cylinders])

# any python generator
train_input_fn = pandas_input_fn(x=input_data, y=input_label,
                                 batch_size=64, shuffle=True,
                                 num_epochs=None)

regressor.train(train_input_fn, steps=10000)




def expirement_fn(run_config, hparams):
    regressor = DNNRegressor(..., config=run_config, hidden_units=hparams['units'])

    return Experiment(estimator=regressor,
                      train_input_fn=pandas_input_fn(...),
                      eval_input_fn=pandas_input_fn(...))

if __name__ == '__main__':
    learn_runner.run(expirement_fn, )