from torch.utils.data import DataLoader
from torchvision import transforms

from .datasets import ImageNetDataset
from .transforms import TwoCropsTransform, GaussianBlur
from .sampler import build_sampler
from .metrics import build_evaluator


def build_common_augmentation(aug_type):
    """
    common augmentation settings for training/testing ImageNet
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if aug_type == 'STANDARD':
        augmentation = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            normalize,
        ]
    elif aug_type == 'AUTOAUG':
        augmentation = [
            transforms.RandomResizedCrop(224),
            ImageNetPolicy(),
            transforms.ToTensor(),
            normalize,
        ]
    elif aug_type == 'MOCOV1':
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    elif aug_type == 'MOCOV2' or aug_type == 'SIMCLR':
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    elif aug_type == 'LINEAR':
        augmentation = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    elif aug_type == 'ONECROP':
        augmentation = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        raise RuntimeError("undefined augmentation type for ImageNet!")

    if aug_type in ['MOCOV1', 'MOCOV2', 'SIMCLR']:
        return TwoCropsTransform(transforms.Compose(augmentation))
    else:
        return transforms.Compose(augmentation)


def build_imagenet_train_dataloader(cfg_dataset, data_type='train'):
    """
    build training dataloader for ImageNet
    """
    cfg_train = cfg_dataset['train']
    # build dataset
    image_reader = cfg_dataset[data_type].get('image_reader', {})
    # PyTorch data preprocessing

    transformer = build_common_augmentation(cfg_train['transforms']['type'])
    dataset = ImageNetDataset(
        root_dir=cfg_train['root_dir'],
        meta_file=cfg_train['meta_file'],
        transform=transformer,
        read_from=cfg_dataset['read_from'],
        image_reader_type=image_reader.get('type', 'pil'),
        server_cfg=cfg_train.get("server", {}),
    )
    # build sampler
    cfg_train['sampler']['kwargs'] = {}
    cfg_dataset['dataset'] = dataset
    sampler = build_sampler(cfg_train['sampler'], cfg_dataset)
    if cfg_dataset['last_iter'] >= cfg_dataset['max_iter']:
        return {'loader': None}

    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg_dataset['batch_size'],
        shuffle=False,
        num_workers=cfg_dataset['num_workers'],
        pin_memory=True,
        sampler=sampler
    )
    return {'type': 'train', 'loader': loader}


def build_imagenet_test_dataloader(cfg_dataset, data_type='test'):
    """
    build testing/validation dataloader for ImageNet
    """
    cfg_test = cfg_dataset['test']
    # build evaluator
    evaluator = None
    if cfg_test.get('evaluator', None):
        evaluator = build_evaluator(cfg_test['evaluator'])
    image_reader = cfg_dataset[data_type].get('image_reader', {})
    # PyTorch data preprocessing
    transformer = build_common_augmentation(cfg_test['transforms']['type'])
    dataset = ImageNetDataset(
        root_dir=cfg_test['root_dir'],
        meta_file=cfg_test['meta_file'],
        transform=transformer,
        read_from=cfg_dataset['read_from'],
        evaluator=evaluator,
        image_reader_type=image_reader.get('type', 'pil'),
    )
    # build sampler
    assert cfg_test['sampler'].get('type', 'distributed') == 'distributed'
    cfg_test['sampler']['kwargs'] = {'dataset': dataset, 'round_up': False}
    cfg_dataset['dataset'] = dataset
    sampler = build_sampler(cfg_test['sampler'], cfg_dataset)
    # PyTorch dataloader
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg_dataset['batch_size'],
        shuffle=False,
        num_workers=cfg_dataset['num_workers'],
        pin_memory=cfg_dataset['pin_memory'],
        sampler=sampler
    )
    return {'type': 'test', 'loader': loader}


def build_imagenet_search_dataloader(cfg_dataset, data_type='arch'):
    """
    build ImageNet dataloader for neural network search (NAS)
    """
    cfg_search = cfg_dataset[data_type]
    # build dataset

    image_reader = cfg_dataset[data_type].get('image_reader', {})
    # PyTorch data preprocessing
    transformer = build_common_augmentation(cfg_search['transforms']['type'])
    dataset = ImageNetDataset(
        root_dir=cfg_search['root_dir'],
        meta_file=cfg_search['meta_file'],
        transform=transformer,
        read_from=cfg_dataset['read_from'],
        image_reader_type=image_reader.get('type', 'pil'),
    )

    # build sampler
    assert cfg_search['sampler'].get('type', 'distributed_iteration') == 'distributed_iteration'
    cfg_search['sampler']['kwargs'] = {}
    cfg_dataset['dataset'] = dataset
    sampler = build_sampler(cfg_search['sampler'], cfg_dataset)
    if cfg_dataset['last_iter'] >= cfg_dataset['max_iter']:
        return {'loader': None}
    # PyTorch dataloader
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg_dataset['batch_size'],
        shuffle=False,
        num_workers=cfg_dataset['num_workers'],
        pin_memory=cfg_dataset['pin_memory'],
        sampler=sampler
    )
    return {'type': data_type, 'loader': loader}
