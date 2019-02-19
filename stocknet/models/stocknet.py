
import keras 


def stocknet(
    inputs, 
    backbone_layers, 
    name = 'stocknet'):

    outputs = backbone_layers 
    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)

