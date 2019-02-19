import tensorflow as tf 
import keras 
from keras.layers import Conv1D, MaxPooling1D, Input, Flatten
from keras.layers import BatchNormalization, GaussianNoise 
from keras.layers import Dense, Dropout, Activation 
from keras.layers import Concatenate, UpSampling1D, Add
from keras.layers import Lambda
from keras import backend as K

import os
import sys

if __name__ == '__main__':
    from backbone import Backbone
    from stocknet import stocknet 
else:
    from .backbone import Backbone
    from .stocknet import stocknet 
    from ..data.dataset import Dataset 


class ResNetBackbone(Backbone):

    def __init__(self, backbone='resnet'):
        super(ResNetBackbone, self).__init__(backbone)

    def model_stocknet(self, weights=True, *args, **kwargs):
        self.model = resnet_stocknet(*args, backbone=self.backbone, **kwargs) 

        if weights:
            self.load_weights()

        return self.model

    def train_stocknet(self, dataset, show_plot=True, save_weights=True, load_weights=False):
        model = self.model_stocknet(weights=load_weights) 

        self.history = model.fit(
            dataset.x_train, dataset.y_train, 
            validation_data=(dataset.x_test, dataset.y_test), 
            epochs=200, 
            batch_size=128
        )
        print('Training {} completed'.format(self.backbone))

        if save_weights:
            self.save_weights()

        if show_plot:
            self.show_history() 
        return 


def resnet_stocknet(backbone, inputs=None, modifier=None, **kwargs):

    if inputs is None:
        inputs = Input(shape=(60, 9))

    net = conv_bn_activation(inputs, 128, 2)
    net = conv_bn_activation(net, 128, 2)
    net = conv_bn_activation(net, 256, 3, strides=2)

    net = residual_block(net, 256)
    net = residual_block(net, 256)
    net = residual_block(net, 256)
    net = residual_block(net, 256)

    net = residual_block(net, 384, kernel_size=3, strides=1)
    net = residual_block(net, 384)
    net = residual_block(net, 384)
    net = residual_block(net, 384)

    net = residual_block(net, 512, kernel_size=3, strides=1)
    net = residual_block(net, 512)
    net = residual_block(net, 512)
    net = residual_block(net, 512)

    net = Flatten()(net)
    net = GaussianNoise(0.1)(net)
    net = dense_pre_activation(net, 1024)
    net = Dropout(0.5)(net)
    net = dense_pre_activation(net, 1024)
    net = Dropout(0.5)(net) 

    net = Dense(3, activation='linear')(net)
    outputs = net

    model = stocknet(inputs=inputs, backbone_layers=outputs, **kwargs)
    # model.compile(optimizer='adadelta', loss='mse', metrics=['mae'])
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model 


def residual_block(inputs, channels, kernel_size=3, strides=1):
    depth = inputs.shape[2].value
    res = conv_pre_activation(inputs, depth, kernel_size, strides)
    net = ResizeBilinear1D(inputs, res)
    net = Concatenate()([net, res])
    net = conv_pre_activation(net, channels, 1)
    print('residual block, output: ', net.shape)
    return net 


def conv_bn_activation(inputs, filters, kernel_size, strides=1, padding='valid', activation='elu'):
    net = Conv1D(filters, kernel_size, strides=strides, padding=padding)(inputs)
    net = BatchNormalization()(net)
    net = Activation(activation=activation)(net)
    return net 


def dense_bn_activation(inputs, units, activation='elu'):
    net = Dense(units)(inputs)
    net = BatchNormalization()(net)
    net = Activation(activation=activation)(net)
    return net 


def dense_pre_activation(inputs, units, activation='elu'):
    net = BatchNormalization()(inputs)
    net = Activation(activation=activation)(net)
    net = Dense(units)(net)
    return net 


def conv_pre_activation(inputs, filters, kernel_size, strides=1, padding='valid', activation='elu'):
    net = BatchNormalization()(inputs)
    net = Activation(activation=activation)(net)
    net = Conv1D(filters, kernel_size, strides=strides, padding=padding)(net)
    return net 


def ResizeBilinear1D(inputs, referer):
    # All backend functions should be called within Lambda() to be as a learnable layer. 
    def resize_like(inputs, referer):
        input_tensor = tf.expand_dims(inputs, axis=1)
        refer_tensor = tf.expand_dims(referer, axis=1)
        h, w = refer_tensor.get_shape()[1], refer_tensor.get_shape()[2] 
        resized = tf.image.resize_bilinear(input_tensor, [h.value, w.value], align_corners=True)
        resized = tf.squeeze(resized, axis=1)
        return resized 

    output_tensor = Lambda(resize_like, arguments={'referer': referer})(inputs)
    return output_tensor


if __name__ == '__main__':
    resnet_stocknet("resnet")
    