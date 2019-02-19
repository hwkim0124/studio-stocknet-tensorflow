import keras 
from keras.layers import Conv1D, MaxPooling1D, Input, Flatten
from keras.layers import BatchNormalization, GaussianNoise 
from keras.layers import Dense, Dropout, Activation 

import os
import sys

# # Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is '':
    path = os.path.join(os.path.dirname(__file__), '..', '..')
    sys.path.insert(0, path)
    import stocknet.models  # noqa: F401
    __package__ = "stocknet.models"


from .backbone import Backbone
from .stocknet import stocknet 
from ..data.dataset import Dataset 


class CnnNetBackbone(Backbone):
    
    def __init__(self, backbone='cnnnet'):
        super(CnnNetBackbone, self).__init__(backbone)

    def model_stocknet(self, weights=True, *args, **kwargs):
        self.model = cnnnet_stocknet(*args, backbone=self.backbone, **kwargs) 

        if weights:
            self.load_weights()

        return self.model

    def train_stocknet(self, dataset, show_plot=True, save_weights=True, load_weights=False):
        model = self.model_stocknet(weights=load_weights) 

        self.history = model.fit(
            dataset.x_train, dataset.y_train, 
            validation_data=(dataset.x_test, dataset.y_test), 
            epochs=100, 
            batch_size=128
        )
        print('Training {} completed'.format(self.backbone))

        if save_weights:
            self.save_weights()

        if show_plot:
            self.show_history() 
        return 


def cnnnet_stocknet(backbone, inputs=None, modifier=None, **kwargs):

    if inputs is None:
        inputs = Input(shape=(60, 9))

    net = conv_bn_activation(inputs, 128, 2)
    net = conv_bn_activation(net, 128, 2)
    net = conv_bn_activation(net, 256, 3, strides=2)

    # net = MaxPooling1D(pool_size=2)(net)

    net = Flatten()(net)
    net = GaussianNoise(0.1)(net)
    net = dense_bn_activation(net, 512)
    net = Dropout(0.5)(net)
    net = dense_bn_activation(net, 512)
    net = Dropout(0.5)(net) 

    net = Dense(3, activation='linear')(net)
    outputs = net

    # A function handler which can modify the backbone before using it in stocknet. 
    # if modifier:
    #     model = modifier(model)

    model = stocknet(inputs=inputs, backbone_layers=outputs, **kwargs)
    model.compile(optimizer='adadelta', loss='mse', metrics=['mae'])
    return model 


def conv_bn_activation(inputs, filters, kernel_size, strides=1, activation='elu'):
    net = Conv1D(filters, kernel_size, strides=strides)(inputs)
    net = BatchNormalization()(net)
    net = Activation(activation=activation)(net)
    return net 


def dense_bn_activation(inputs, units, activation='elu'):
    net = Dense(units)(inputs)
    net = BatchNormalization()(net)
    net = Activation(activation=activation)(net)
    return net 


if __name__ == '__main__':
    b = CnnNetBackbone() 
    model = b.model_stocknet(weights=False) 
    print(model.output_shape)
    b.save_weights()