from .imagenet_evaluator import ImageNetEvaluator


def build_evaluator(cfg):
    evaluator = {
        'imagenet': ImageNetEvaluator,
    }[cfg['type']]
    return evaluator(**cfg['kwargs'])
