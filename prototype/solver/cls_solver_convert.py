import os
import argparse
from easydict import EasyDict
from tensorboardX import SummaryWriter
import pprint
import torch

from .base_solver import BaseSolver
from prototype.utils.dist import link_dist, DistModule, broadcast_object
from prototype.utils.misc import makedir, create_logger, get_logger, count_params, count_flops, \
    param_group_all, AverageMeter, accuracy, load_state_model, load_state_optimizer, mixup_data, \
    mix_criterion, modify_state, cutmix_data, parse_config
from prototype.utils.ema import EMA
from prototype.model import model_entry
from prototype.optimizer import optim_entry, FP16RMSprop, FP16SGD, FusedFP16SGD, FP16AdamW
from prototype.lr_scheduler import scheduler_entry
from prototype.data import build_imagenet_train_dataloader, build_imagenet_test_dataloader
from prototype.data import build_custom_dataloader

import onnx
import spring.linklink as link
from spring.nart.utils.onnx_utils import OnnxDecoder
from spring.nart.passes import DeadCodeElimination, ConvFuser, GemmFuser
from spring.nart.core import Model
import spring.nart.tools.pytorch as pytorch
from spring.utils.log_helper import default_logger


class ClsSolver(BaseSolver):

    def __init__(self, config_file):
        self.config_file = config_file
        self.prototype_info = EasyDict()
        self.config = parse_config(config_file)
        self.setup_env()
        self.build_model()
        self.build_optimizer()
        self.build_data()
        self.build_lr_scheduler()

    def setup_env(self):
        # dist
        self.dist = EasyDict()
        self.dist.rank, self.dist.world_size = link.get_rank(), link.get_world_size()
        self.prototype_info.world_size = self.dist.world_size
        # directories
        self.path = EasyDict()
        self.path.root_path = os.path.dirname(self.config_file)
        self.path.save_path = os.path.join(self.path.root_path, 'checkpoints')
        self.path.event_path = os.path.join(self.path.root_path, 'events')
        self.path.result_path = os.path.join(self.path.root_path, 'results')
        makedir(self.path.save_path)
        makedir(self.path.event_path)
        makedir(self.path.result_path)
        # tb_logger
        if self.dist.rank == 0:
            self.tb_logger = SummaryWriter(self.path.event_path)
        # logger
        create_logger(os.path.join(self.path.root_path, 'log.txt'))
        self.logger = get_logger(__name__)
        self.logger.info(f'config: {pprint.pformat(self.config)}')
        if 'SLURM_NODELIST' in os.environ:
            self.logger.info(f"hostnames: {os.environ['SLURM_NODELIST']}")
        # load pretrain checkpoint
        if hasattr(self.config.saver, 'pretrain'):
            self.state = torch.load(self.config.saver.pretrain.path, 'cpu')
            self.logger.info(f"Recovering from {self.config.saver.pretrain.path}, keys={list(self.state.keys())}")
            if hasattr(self.config.saver.pretrain, 'ignore'):
                self.state = modify_state(self.state, self.config.saver.pretrain.ignore)
        else:
            self.state = {}
            self.state['last_iter'] = 0
        # others
        torch.backends.cudnn.benchmark = True

    def build_model(self):
        if hasattr(self.config, 'lms'):
            if self.config.lms.enable:
                torch.cuda.set_enabled_lms(True)
                byte_limit = self.config.lms.kwargs.limit * (1 << 30)
                torch.cuda.set_limit_lms(byte_limit)
                self.logger.info('Enable large model support, limit of {}G!'.format(
                    self.config.lms.kwargs.limit))

        self.model = model_entry(self.config.model)
        self.prototype_info.model = self.config.model.type
        self.model.cuda()

        count_params(self.model)
        count_flops(self.model, input_shape=[
                    1, 3, self.config.data.input_size, self.config.data.input_size])

        # handle fp16
        if self.config.optimizer.type == 'FP16SGD' or \
           self.config.optimizer.type == 'FusedFP16SGD' or \
           self.config.optimizer.type == 'FP16RMSprop' or \
           self.config.optimizer.type == 'FP16AdamW':
            self.fp16 = True
        else:
            self.fp16 = False

        if self.fp16:
            # if you have modules that must use fp32 parameters, and need fp32 input
            # try use link.fp16.register_float_module(your_module)
            # if you only need fp32 parameters set cast_args=False when call this
            # function, then call link.fp16.init() before call model.half()
            if self.config.optimizer.get('fp16_normal_bn', False):
                self.logger.info('using normal bn for fp16')
                link.fp16.register_float_module(link.nn.SyncBatchNorm2d, cast_args=False)
                link.fp16.register_float_module(torch.nn.BatchNorm2d, cast_args=False)
            if self.config.optimizer.get('fp16_normal_fc', False):
                self.logger.info('using normal fc for fp16')
                link.fp16.register_float_module(torch.nn.Linear, cast_args=True)
            link.fp16.init()
            self.model.half()

        self.model = DistModule(self.model, self.config.dist.sync)

        if 'model' in self.state:
            load_state_model(self.model, self.state['model'])

    def build_optimizer(self):

        opt_config = self.config.optimizer
        opt_config.kwargs.lr = self.config.lr_scheduler.kwargs.base_lr
        self.prototype_info.optimizer = self.config.optimizer.type

        # make param_groups
        pconfig = {}

        if opt_config.get('no_wd', False):
            pconfig['conv_b'] = {'weight_decay': 0.0}
            pconfig['linear_b'] = {'weight_decay': 0.0}
            pconfig['bn_w'] = {'weight_decay': 0.0}
            pconfig['bn_b'] = {'weight_decay': 0.0}

        if 'pconfig' in opt_config:
            pconfig.update(opt_config['pconfig'])

        param_group, type2num = param_group_all(self.model, pconfig)

        opt_config.kwargs.params = param_group

        self.optimizer = optim_entry(opt_config)

        if 'optimizer' in self.state:
            load_state_optimizer(self.optimizer, self.state['optimizer'])

        # EMA
        if self.config.ema.enable:
            self.config.ema.kwargs.model = self.model
            self.ema = EMA(**self.config.ema.kwargs)
        else:
            self.ema = None

        if 'ema' in self.state:
            self.ema.load_state_dict(self.state['ema'])

    def build_lr_scheduler(self):
        self.prototype_info.lr_scheduler = self.config.lr_scheduler.type
        if not getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.lr_scheduler.kwargs.max_iter = self.config.data.max_iter
        self.config.lr_scheduler.kwargs.optimizer = self.optimizer.optimizer if isinstance(self.optimizer, FP16SGD) or \
            isinstance(self.optimizer, FP16RMSprop) or isinstance(self.optimizer, FP16AdamW) else self.optimizer
        self.config.lr_scheduler.kwargs.last_iter = self.state['last_iter']
        self.lr_scheduler = scheduler_entry(self.config.lr_scheduler)

    def build_data(self):
        self.config.data.last_iter = self.state['last_iter']
        if getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.data.max_iter = self.config.lr_scheduler.kwargs.max_iter
        else:
            self.config.data.max_epoch = self.config.lr_scheduler.kwargs.max_epoch

        if self.config.data.get('type', 'imagenet') == 'imagenet':
            self.train_data = build_imagenet_train_dataloader(self.config.data)
        else:
            self.train_data = build_custom_dataloader('train', self.config.data)

        if self.config.data.get('type', 'imagenet') == 'imagenet':
            self.val_data = build_imagenet_test_dataloader(self.config.data)
        else:
            self.val_data = build_custom_dataloader('test', self.config.data)


    def merge_onnx_model(self, model, onnx_name):
        merged_onnx_name = onnx_name.replace('.onnx', '_merged.onnx')
        if link.get_rank() == 0:
            self.logger.info("merge model...")
            onnx_model = onnx.load(onnx_name)

            graph = OnnxDecoder().decode(onnx_model)
            graph.update_tensor_shape()
            graph.update_topology()

            ConvFuser().run(graph)
            GemmFuser().run(graph)
            DeadCodeElimination().run(graph)
            graph.update_topology()

            onnx_model = Model.make_model(graph)
            onnx_model = onnx_model.dump_to_onnx()

            onnx.save(onnx_model, merged_onnx_name)
        link.barrier()
        model.train()
        model.cuda()

        return merged_onnx_name


    def tocaffe(self,
                model,
                image_size,
                save_prefix,
                input_names=['data'],
                output_names=['output'],
                onnx_only=True):
        data_shape = [image_size]
        onnx_name = save_prefix + '.onnx'
        prototxt_name = save_prefix + '.prototxt'
        caffemodel_name = save_prefix + '.caffemodel'

        if link.get_rank() == 0:
            if onnx_only:
                with pytorch.convert_mode():
                    pytorch.export_onnx(model,
                                        data_shape,
                                        filename=save_prefix,
                                        input_names=input_names,
                                        output_names=output_names,
                                        verbose=False,
                                        cloze=False)
            else:
                with pytorch.convert_mode():
                    pytorch.convert_v2(model,
                                       data_shape,
                                       filename=save_prefix,
                                       input_names=input_names,
                                       output_names=output_names,
                                       verbose=False,
                                       cloze=False)
                from spring.nart.tools.caffe.count import countFile
                _, info = countFile(prototxt_name)
                info_dict = {}
                info_dict['name'], info_dict['inputData'], info_dict['param'], \
                info_dict['activation'], info_dict['comp'], info_dict['add'], \
                info_dict['div'], info_dict['macc'], info_dict['exp'], info_dict['memacc'] = info
                for k, v in info_dict.items():
                    if not isinstance(v, str):
                        v = v / 1e6
                        info_dict[k] = v
                default_logger.info(info_dict)
                os.system('python -m spring.nart.caffe2onnx {} {} -o {}'.format(
                    prototxt_name, caffemodel_name, onnx_name))

        merged_onnx_name = self.merge_onnx_model(model, onnx_name)

        return merged_onnx_name

    def convert(self):
        self.tocaffe(self.model, image_size=[1, 3, 224, 224], save_prefix=self.config.model['type'])


@link_dist
def main():
    parser = argparse.ArgumentParser(description='Classification Solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()
    # build solver
    solver = ClsSolver(args.config)
    solver.convert()
    solver.logger.info('Convert Model To Onnx!')


if __name__ == '__main__':
    main()
