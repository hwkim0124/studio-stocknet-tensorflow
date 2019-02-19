
from .cnnnet import CnnNetBackbone 
from .resnet import ResNetBackbone


def backbone(backbone_name):
    if 'cnnnet' in backbone_name:
        return CnnNetBackbone(backbone_name)
    elif 'resnet' in backbone_name:
        return ResNetBackbone(backbone_name)
    else:
        raise NotImplementedError('Backbone class for \'{}\' not implemented.'.format(backbone_name))
    
   