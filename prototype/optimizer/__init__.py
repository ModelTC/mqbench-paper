from torch.optim import SGD, RMSprop, Adadelta, Adagrad, Adam, AdamW  # noqa F401


def optim_entry(config):
    return globals()[config['type']](**config['kwargs'])
