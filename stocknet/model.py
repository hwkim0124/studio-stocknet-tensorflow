
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation 
from keras.layers import Conv1D, MaxPooling1D, Input, Flatten
from keras.layers import BatchNormalization, GaussianNoise 
from keras.optimizers import Adam

import numpy as np 
import math 

from data import load_dataset


def base_cnn_mlp_regr(): 
    model = Sequential() 
    # input : 1 x 30 data images with 9 channels -> (30, 9) tensors 
    model.add(Conv1D(128, 2, use_bias='false', input_shape=(30, 9)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Conv1D(64, 1, use_bias='false'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(Conv1D(128, 2, use_bias='false'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Conv1D(64, 1, use_bias='false'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(Conv1D(256, 3, use_bias='false'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Conv1D(128, 1, use_bias='false'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(MaxPooling1D(pool_size=2)) 

    model.add(Flatten())
    model.add(GaussianNoise(0.1))
    # model.add(Dropout(0.25))
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.5))

    # model.add(Dense(51, activation='softmax'))
    model.add(Dense(1, activation='linear'))

    # model.compile(optimizer='adadelta', loss='categorical_crossentropy')
    model.compile(optimizer='adadelta', loss='mse', metrics=['mae'])
    return model 


# x_train = np.random.random((1000, 30, 9))
# y_train = np.random.random((1000, 3))

# dset = load_dataset()
# x_train = dset['x_train'] 
# y_train = dset['y_train'] 
# x_test = dset['x_test'] 
# y_test = dset['y_test'] 

# y_train2 = y_train[:, 0]
# y_test2 = y_test[:, 0]

# y_train2 = np.clip(np.ceil(y_train2 * 100.0 - 0.5), -25, +25) 
# y_eval2 = np.clip(np.ceil(y_eval2 * 100.0 - 0.5), -25, +25) 

# unique, counts = np.unique(y_train2, return_counts=True)
# hist = dict(zip(unique, counts))
# print(hist)

# y_train3 = keras.utils.to_categorical(y_train2 + 25, num_classes=51)
# y_eval3 = keras.utils.to_categorical(y_eval2 + 25, num_classes=51)

# model.fit(x_train, y_train2, 
#             epochs=100, batch_size=256)

# model.save_weights('weights.h5')
# # model.load_weights('weights.h5')

# # score = model.evaluate(x_eval, y_eval3)
# # print('Test loss: ', score)

# predictions = model.predict_proba(x_eval)

# count = 0 
# correct = 0 
# for i in range(len(y_eval2)):
#     dist = np.argsort(-predictions[i]) 
#     # print('Target: %d' % (int(y_eval2[i]) + 25))

#     # for k in range(5):
#     #     print('%d (%.2f)' % (dist[k], predictions[i][dist[k]]))
#     target = int(y_eval2[i] + 25)
#     output = predictions[i][dist[0]]

#     if target != 25: 
#         count += 1 
#         if (target < 25 and output < 25) or (target > 25 and output > 25):
#             correct += 1 

# print('Total predictions: %d, correct: %d' % (count, correct))

# a = 1 







        