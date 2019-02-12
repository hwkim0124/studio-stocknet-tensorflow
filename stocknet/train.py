
import matplotlib.pyplot as plt
import numpy as np 

import model 
from data import load_dataset

def evaluate_network():
    dset = load_dataset()
    x_train = dset['x_train'] 
    y_train = dset['y_train'] 
    x_test = dset['x_test'] 
    y_test = dset['y_test'] 

    y_train2 = y_train[:, 0]
    y_test2 = y_test[:, 0]

    network = model.base_cnn_mlp_regr() 
    network.load_weights('weights.h5')

    train_metrics = network.evaluate(x_train, y_train2)
    test_metrics = network.evaluate(x_test, y_test2)
    print('Train loss: ', train_metrics[0], 'test loss: ', test_metrics[0])

    predicts = network.predict(x_train)
    corrects = 0 
    size = len(x_train) 
    for i in range(size):
        # print('%.2f : %.2f' % (pred, y_train2[i]))
        if predicts[i] * y_train2[i] > 0: 
            corrects += 1 

    print('Train examples: %d, corrects: %d (%.2f)' % (size, corrects, (corrects/size)))

    predicts = network.predict(x_test)
    corrects = 0 
    size = len(x_test) 
    for i in range(size):
        # print('%.2f : %.2f' % (predicts[i], y_test2[i]))
        if predicts[i] * y_test2[i] > 0: 
            corrects += 1 

    print('Test examples: %d, corrects: %d (%.2f)' % (size, corrects, (corrects/size)))


def train_network(): 
    dset = load_dataset()
    x_train = dset['x_train'] 
    y_train = dset['y_train'] 
    x_test = dset['x_test'] 
    y_test = dset['y_test'] 

    y_train2 = y_train[:, 0]
    y_test2 = y_test[:, 0]

    network = model.base_cnn_mlp_regr() 
    history = network.fit(x_train, y_train2, validation_data=(x_test, y_test2),
                            epochs=500, batch_size=512)
    network.save_weights('weights.h5')

    np.save('train_loss.npy', history.history['loss'])
    np.save('test_loss.npy', history.history['val_loss'])
    np.save('train_mae.npy', history.history['mean_absolute_error'])
    np.save('test_mae.npy', history.history['val_mean_absolute_error'])

    # plot history
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='test_loss')
    plt.plot(history.history['mean_absolute_error'], label='train_mae')
    plt.plot(history.history['val_mean_absolute_error'], label='test_mae')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    #train_network()
    evaluate_network()