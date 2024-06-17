import pandas as pd
import tensorflow as tf

traindf = pd.read_csv('UNSW_NB15_training-set.csv')
testdf = pd.read_csv('UNSW_NB15_testing-set.csv')

# print cols
print(list(traindf))

# print type of value in each column
print(traindf.dtypes)

# print cols with object type
print(traindf.select_dtypes(include=['object']))

# # print unique values in attack type column
# print(traindf['attack_cat'].unique())

# # remove all the rows if attack_cat is not normal or DoS
# traindf = traindf[traindf['attack_cat'].isin(['Normal', 'DoS'])]
# testdf = testdf[testdf['attack_cat'].isin(['Normal', 'DoS'])]

# # remove cols with object type
# traindf = traindf.select_dtypes(exclude=['object'])
# testdf = testdf.select_dtypes(exclude=['object'])


# # drop nas
# traindf = traindf.dropna()
# testdf = testdf.dropna()

# # labelling

# # use all col except for label as features
# x_train = traindf.drop(columns=['label'])
# x_test = testdf.drop(columns=['label'])

# # use label as target
# y_train = traindf['label']
# y_test = testdf['label']

# # tensorflow model with label as target
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=10)

# loss_and_metrics = model.evaluate(x_test, y_test)
# print(loss_and_metrics)
# print('Loss = ',loss_and_metrics[0])
# print('Accuracy = ',loss_and_metrics[1])



