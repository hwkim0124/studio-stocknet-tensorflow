
import keras 
import numpy as np 
import matplotlib.pyplot as plt


class Backbone(object):
    def __init__(self, backbone):
        self.backbone = backbone 
        self.model = None 
        self.history = None 
        self.train_metrics = None 
        self.test_metrics = None 
        
    def model_stocknet(self, weights=True, *args, **kwargs):
        raise NotImplementedError('stocknet not implemented')

    def load_weights(self):
        if self.model is not None:
            self.model.load_weights('./weights/{}_weights.h5'.format(self.backbone))

    def save_weights(self):
        if self.model is not None:
            self.model.save_weights('./weights/{}_weights.h5'.format(self.backbone))

            if self.history is not None:
                np.save('./weights/{}_train_loss.npy'.format(self.backbone), self.history.history['loss'])
                np.save('./weights/{}_test_loss.npy'.format(self.backbone), self.history.history['val_loss'])
                np.save('./weights/{}_train_mae.npy'.format(self.backbone), self.history.history['mean_absolute_error'])
                np.save('./weights/{}_test_mae.npy'.format(self.backbone), self.history.history['val_mean_absolute_error'])

    def show_history(self):
        if self.history is not None:
            plt.plot(self.history.history['loss'], label='train_loss')
            plt.plot(self.history.history['val_loss'], label='test_loss')
            plt.plot(self.history.history['mean_absolute_error'], label='train_mae')
            plt.plot(self.history.history['val_mean_absolute_error'], label='test_mae')
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test', 'Train MAE.', 'Test MAE.'], loc='upper left')
            plt.show()
        return

    def evaluate_stocknet(self, dataset):   
        model = self.model_stocknet() 

        self.train_metrics = model.evaluate(dataset.x_train, dataset.y_train)
        self.test_metrics = model.evaluate(dataset.x_test, dataset.y_test)
        print('Train dataset loss: ', self.train_metrics[0], 'test dataset loss: ', self.test_metrics[0])
        return 

    def predict_stocknet(self, dataset, train_data=False, test_data=True):
        model = self.model_stocknet() 

        if train_data:
            predicts = model.predict(dataset.x_train)
            agrees1 = 0 
            agrees2 = 0
            size = dataset.train_size() 

            for i in range(size):
                # print('{} : Test data: '.format(i), dataset.y_test[i], ', predict: ', predicts[i])
                last_close = dataset.x_train[i, 59, 3]
                pred_change = predicts[i, 0] - last_close
                real_change = dataset.y_train[i, 0] - last_close

                if abs(pred_change - real_change) < 0.01 or pred_change * real_change > 0:
                    agrees1 += 1 

                pred_change = predicts[i, 2] - last_close
                real_change = dataset.y_train[i, 2] - last_close

                if abs(pred_change - real_change) < 0.01 or pred_change * real_change > 0:
                    agrees2 += 1 

            print('Train examples: %d, corrects: %d (%.4f), %d (%.4f)' % (size, agrees1, (agrees1/size), agrees2, (agrees2/size)))

        if test_data:
            predicts = model.predict(dataset.x_test)
            agrees1 = 0 
            agrees2 = 0
            size = dataset.test_size() 

            for i in range(size):
                # print('{} : Test data: '.format(i), dataset.y_test[i], ', predict: ', predicts[i])
                last_close = dataset.x_test[i, 59, 3]
                pred_change = predicts[i, 0] - last_close
                real_change = dataset.y_test[i, 0] - last_close

                if abs(pred_change - real_change) < 0.01 or pred_change * real_change > 0:
                    agrees1 += 1 

                pred_change = predicts[i, 2] - last_close
                real_change = dataset.y_test[i, 2] - last_close

                if abs(pred_change - real_change) < 0.01 or pred_change * real_change > 0:
                    agrees2 += 1 

            print('Test examples: %d, corrects: %d (%.4f), %d (%.4f)' % (size, agrees1, (agrees1/size), agrees2, (agrees2/size)))

        return 


