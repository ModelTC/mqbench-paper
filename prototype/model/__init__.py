from .mobilenet_v2 import mobilenetv2 as mobilenet_v2  # noqa: F401
from .regnet import (  # noqa: F401
    regnetx_200m, regnetx_400m, regnetx_600m, regnetx_800m,
    regnetx_1600m, regnetx_3200m, regnetx_4000m, regnetx_6400m,
    regnety_200m, regnety_400m, regnety_600m, regnety_800m,
    regnety_1600m, regnety_3200m, regnety_4000m, regnety_6400m,
)
from .resnet import (  # noqa: F401
    resnet18, resnet26, resnet34, resnet50,
    resnet101, resnet152, resnet_custom
)
from .efficientnet_lite import (  # noqa: F401
    efficientnetlite_b0
 )

from .mnasnet import mnasnet  # noqa: F401


import torch
from collections import OrderedDict

load_path = {
    'resnet18': '/mnt/lustre/liyuhang1/ImageNet/ckpts/resnet18_fp_strongbaseline/ckpt.pth.tar',
    'resnet50': '/mnt/lustre/liyuhang1/ImageNet/ckpts/resnet50_fp_strongbaseline/ckpt.pth.tar',
    'regnetx_600m': '/mnt/lustre/liyuhang1/ImageNet/ckpts/regnet_600m.pth.tar',
    'mobilenet_v2': '/mnt/lustre/liyuhang1/ImageNet/ckpts/mobilenetv2.tar',
    'efficientnetlite_b0': '/mnt/lustre/liyuhang1/ImageNet/ckpts/efficientnet_lite0_ra-37913777.pth'
}


def load_model_pytorch(model, load_model, replace_dict={}):

    checkpoint = torch.load(load_model, map_location='cpu')

    if 'state_dict' in checkpoint.keys():
        load_from = checkpoint['state_dict']
    elif 'model' in checkpoint.keys():
        load_from = checkpoint['model']
    else:
        load_from = checkpoint

    # remove "module." in case the model is saved with Dist Mode
    if 'module.' in list(load_from.keys())[0]:
        load_from = OrderedDict([(k.replace("module.", ""), v) for k, v in load_from.items()])
    for keys in replace_dict.keys():
        load_from = OrderedDict([(k.replace(keys, replace_dict[keys]), v) for k, v in load_from.items()])

    model.load_state_dict(load_from, strict=False)


def model_entry(config, pretrained=True):

    if config['type'] not in globals():
        if config['type'].startswith('spring_'):
            try:
                from spring.models import SPRING_MODELS_REGISTRY
            except ImportError:
                print('Please install Spring2 first!')
            model_name = config['type'][len('spring_'):]
            config['type'] = model_name
            return SPRING_MODELS_REGISTRY.build(config)
        else:
            from prototype.spring import PrototypeHelper
            return PrototypeHelper.external_model_builder[config['type']](**config['kwargs'])

    model = globals()[config['type']](**config['kwargs'])

    if pretrained:
        load_model_pytorch(model, load_path[config['type']])

    return model
