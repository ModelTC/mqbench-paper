import timm

BN = None

__all__ = ['efficientnetlite_b0', 'efficientnetlite_b1', 'efficientnetlite_b2', 'efficientnetlite_b3',
           'efficientnetlite_b4']


def efficientnetlite_b0(**kwargs):
    """
    Constructs a EfficientNet-B0 model.
    """
    model = timm.create_model('efficientnet_lite0', pretrained=False)

    return model


def efficientnetlite_b1(**kwargs):
    """
    Constructs a EfficientNet-B1 model. No pretrain Models
    """
    del kwargs['bn']
    model = timm.create_model('efficientnet_lite1', pretrained=True, **kwargs)

    return model

def efficientnetlite_b2(**kwargs):
    """
    Constructs a EfficientNet-B2 model. No pretrain Models
    """
    del kwargs['bn']
    model = timm.create_model('efficientnet_lite2', pretrained=True, **kwargs)

    return model


def efficientnetlite_b3(**kwargs):
    """
    Constructs a EfficientNet-B3 model. No pretrain Models
    """
    del kwargs['bn']
    model = timm.create_model('efficientnet_lite3', pretrained=True, **kwargs)

    return model

def efficientnetlite_b4(**kwargs):
    """
    Constructs a EfficientNet-B4 model. No pretrain Models
    """
    del kwargs['bn']
    model = timm.create_model('efficientnet_lite4', pretrained=True, **kwargs)

    return model

