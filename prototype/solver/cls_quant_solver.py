import os
import argparse
from easydict import EasyDict
from tensorboardX import SummaryWriter
import pprint
import random
import time
import datetime
import torch
import json
import spring.linklink as link
import torch.nn.functional as F
import torch.quantization.quantize_fx as quantize_fx
import numpy as np

from .base_solver import BaseSolver
from prototype.utils.dist import link_dist, DistModule, broadcast_object
from prototype.utils.misc import makedir, create_logger, get_logger, param_group_all, AverageMeter, accuracy, \
    load_state_model, load_state_optimizer, modify_state, parse_config
from prototype.model import model_entry
from prototype.optimizer import optim_entry
from prototype.lr_scheduler import scheduler_entry
from prototype.data import build_imagenet_train_dataloader, build_imagenet_test_dataloader
from prototype.loss_functions import LabelSmoothCELoss
from prototype.quantization.prepare_quantization import prepare_quant_academic, enable_param_learning, \
    bitwidth_refactor, toggle_fake_quant, get_foldbn_config, get_qconfig
from prototype.quantization.bn_fold import freeze_bn_all, search_fold_bn


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

    def seed_all(self, seed=1000):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

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
            try:
                self.state = torch.load(self.config.saver.pretrain.path, 'cpu')
                self.logger.info(f"Recovering from {self.config.saver.pretrain.path}, keys={list(self.state.keys())}")
                if hasattr(self.config.saver.pretrain, 'ignore'):
                    self.state = modify_state(self.state, self.config.saver.pretrain.ignore)
            except:
                self.state = {}
                self.state['last_iter'] = 0
        else:
            self.state = {}
            self.state['last_iter'] = 0
        # others
        self.seed_all(self.config.seed if hasattr(self.config, 'seed') else 1000)

    def build_model(self):

        self.model = model_entry(self.config.model, pretrained=True)
        self.prototype_info.model = self.config.model.type

        self.qparams = self.config.qparams
        backend = self.config.backend
        if hasattr(self.config, 'bnfold'):
            foldbn_strategy = self.config.bnfold
        else:
            foldbn_strategy = 1                    # default use torch fold

        self.freeze_step = self.config.freeze_step if hasattr(self.config, 'freeze_step') else 10000000

        if backend == 'academic':
            backend_params = dict(ada_sign=True, symmetry=True, per_channel=False, pot_scale=False)
            self.model = prepare_quant_academic(self.model, **self.qparams, **backend_params)
        elif backend == 'acl':
            backend_params = dict(ada_sign=False, symmetry=False, per_channel=True, pot_scale=False)
            self.model = prepare_quant_academic(self.model, **self.qparams, **backend_params)
            search_fold_bn(self.model, strategy=foldbn_strategy)
        elif backend == 'bnfold_ablation':
            backend_params = dict(ada_sign=True, symmetry=True, per_channel=False, pot_scale=False)
            self.model = prepare_quant_academic(self.model, **self.qparams, **backend_params)
            search_fold_bn(self.model, strategy=foldbn_strategy)
        elif backend == 'graph_ablation':
            if foldbn_strategy == 1:
                backend_params = dict(ada_sign=True, symmetry=True, per_channel=False, pot_scale=False)
                self.model = prepare_quant_academic(self.model, **self.qparams, **backend_params)
            else:
                backend_params = dict(ada_sign=True, symmetry=True, per_channel=False, pot_scale=False)
                model_qconfig = get_qconfig(**self.qparams, **backend_params)
                foldbn_config = get_foldbn_config(-1)
                self.model = quantize_fx.prepare_qat_fx(self.model, {"": model_qconfig}, foldbn_config)
                bitwidth_refactor(self.model, model_qconfig.activation)
                if foldbn_strategy == 2:
                    from prototype.quantization.prepare_tensorrt import tensorrt_refactor
                    tensorrt_refactor(self.model, self.qparams["a_method"],
                                      foldbn_config['additional_qat_module_mapping'])

        else:
            if backend == 'tvm':
                backend_params = dict(ada_sign=False, symmetry=True, per_channel=False, pot_scale=True)
            elif backend == 'snpe':
                backend_params = dict(ada_sign=False, symmetry=False, per_channel=False, pot_scale=False)
            elif backend == 'fbgemm':
                backend_params = dict(ada_sign=False, symmetry=False, per_channel=True, pot_scale=False)
            elif backend == "tensorrt":
                backend_params = dict(ada_sign=False, symmetry=True, per_channel=True, pot_scale=False)
            else:
                raise NotImplementedError
            model_qconfig = get_qconfig(**self.qparams, **backend_params)
            foldbn_config = get_foldbn_config(foldbn_strategy)
            self.model = quantize_fx.prepare_qat_fx(self.model, {"": model_qconfig}, foldbn_config)
            bitwidth_refactor(self.model, model_qconfig.activation)

            if backend == "tensorrt":
                from prototype.quantization.prepare_tensorrt import tensorrt_refactor
                tensorrt_refactor(self.model, self.qparams["a_method"], foldbn_config['additional_qat_module_mapping'])

        self.model.eval()
        toggle_fake_quant(self.model, enabled=False)
        self.model(torch.randn(1, 3, 224, 224))
        toggle_fake_quant(self.model, enabled=True)
        self.model.cuda()
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

    def build_lr_scheduler(self):
        self.prototype_info.lr_scheduler = self.config.lr_scheduler.type
        if not getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.lr_scheduler.kwargs.max_iter = self.config.data.max_iter
        self.config.lr_scheduler.kwargs.optimizer = self.optimizer
        self.config.lr_scheduler.kwargs.last_iter = self.state['last_iter']
        self.lr_scheduler = scheduler_entry(self.config.lr_scheduler)

    def build_data(self):
        self.config.data.last_iter = self.state['last_iter']
        if getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.data.max_iter = self.config.lr_scheduler.kwargs.max_iter
        else:
            self.config.data.max_epoch = self.config.lr_scheduler.kwargs.max_epoch

        self.train_data = build_imagenet_train_dataloader(self.config.data)
        self.val_data = build_imagenet_test_dataloader(self.config.data)

    def pre_train(self):
        self.meters = EasyDict()
        self.meters.batch_time = AverageMeter(self.config.saver.print_freq)
        self.meters.fwd_time = AverageMeter(self.config.saver.print_freq)
        self.meters.bwd_time = AverageMeter(self.config.saver.print_freq)
        self.meters.step_time = AverageMeter(self.config.saver.print_freq)
        self.meters.data_time = AverageMeter(self.config.saver.print_freq)
        self.meters.losses = AverageMeter(self.config.saver.print_freq)
        self.meters.top1 = AverageMeter(self.config.saver.print_freq)
        self.meters.top5 = AverageMeter(self.config.saver.print_freq)

        self.model.train()

        label_smooth = self.config.get('label_smooth', 0.0)
        self.num_classes = self.config.model.kwargs.get('num_classes', 1000)
        self.topk = 5 if self.num_classes >= 5 else self.num_classes
        if label_smooth > 0:
            self.logger.info('using label_smooth: {}'.format(label_smooth))
            self.criterion = LabelSmoothCELoss(label_smooth, self.num_classes)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def train(self):

        self.pre_train()
        self.best_acc = 0.0
        total_step = len(self.train_data['loader'])
        start_step = self.state['last_iter'] + 1
        end = time.time()
        for i, batch in enumerate(self.train_data['loader']):
            input = batch['image']
            target = batch['label']
            curr_step = start_step + i
            self.lr_scheduler.step(curr_step)
            # lr_scheduler.get_lr()[0] is the main lr
            current_lr = self.lr_scheduler.get_lr()[0]
            # measure data loading time
            self.meters.data_time.update(time.time() - end)
            # transfer input to gpu
            target = target.squeeze().cuda().long()
            input = input.cuda()
            # forward
            if i == 0:
                if curr_step == 1:
                    # LSQ Initialization
                    _ = self.model(input)
                    enable_param_learning(self.model)
                    self.model.zero_grad()
                else:
                    # resume training
                    enable_param_learning(self.model)

            if curr_step >= self.freeze_step:
                freeze_bn_all(self.model)

            time1 = time.time()
            logits = self.model(input)
            self.meters.fwd_time.update(time.time() - time1)

            loss = self.criterion(logits, target) / self.dist.world_size
            # measure accuracy and record loss
            prec1, prec5 = accuracy(logits, target, topk=(1, self.topk))

            reduced_loss = loss.clone()
            reduced_prec1 = prec1.clone() / self.dist.world_size
            reduced_prec5 = prec5.clone() / self.dist.world_size

            self.meters.losses.reduce_update(reduced_loss)
            self.meters.top1.reduce_update(reduced_prec1)
            self.meters.top5.reduce_update(reduced_prec5)

            # compute and update gradient
            self.optimizer.zero_grad()
            time1 = time.time()
            loss.backward()
            self.model.sync_gradients()

            self.meters.bwd_time.update(time.time() - time1)

            self.optimizer.step()
            # measure elapsed time
            self.meters.batch_time.update(time.time() - end)

            # training logger
            if curr_step % self.config.saver.print_freq == 0 and self.dist.rank == 0:
                self.tb_logger.add_scalar('loss_train', self.meters.losses.avg, curr_step)
                self.tb_logger.add_scalar('acc1_train', self.meters.top1.avg, curr_step)
                self.tb_logger.add_scalar('acc5_train', self.meters.top5.avg, curr_step)
                self.tb_logger.add_scalar('lr', current_lr, curr_step)
                remain_secs = (total_step - curr_step) * self.meters.batch_time.avg
                remain_time = datetime.timedelta(seconds=round(remain_secs))
                finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()+remain_secs))
                log_msg = f'Iter: [{curr_step}/{total_step}]\t' \
                    f'Time {self.meters.batch_time.val:.3f} ({self.meters.batch_time.avg:.3f})\t' \
                    f'Fwd Time {self.meters.fwd_time.val:.3f} ({self.meters.fwd_time.avg:.3f})\t' \
                    f'Bwd Time {self.meters.bwd_time.val:.3f} ({self.meters.bwd_time.avg:.3f})\t' \
                    f'Data {self.meters.data_time.val:.3f} ({self.meters.data_time.avg:.3f})\t' \
                    f'Loss {self.meters.losses.val:.4f} ({self.meters.losses.avg:.4f})\t' \
                    f'Prec@1 {self.meters.top1.val:.3f} ({self.meters.top1.avg:.3f})\t' \
                    f'Prec@5 {self.meters.top5.val:.3f} ({self.meters.top5.avg:.3f})\t' \
                    f'LR {current_lr:.4f}\t' \
                    f'Remaining Time {remain_time} ({finish_time})'
                self.logger.info(log_msg)

            # testing during training
            if curr_step > 0 and curr_step % self.config.saver.val_freq == 0:
                metrics = self.evaluate()

                # testing logger
                if self.dist.rank == 0 and self.config.data.test.evaluator.type == 'imagenet':
                    metric_key = 'top{}'.format(self.topk)
                    self.tb_logger.add_scalar('acc1_val', metrics.metric['top1'], curr_step)
                    self.tb_logger.add_scalar('acc5_val', metrics.metric[metric_key], curr_step)
                # determine best acc
                if metrics.metric['top1'] > self.best_acc:
                    self.best_acc = metrics.metric['top1']
                self.logger.info('Best top1 accuracy: {}'.format(self.best_acc))
                # save ckpt
                if self.dist.rank == 0:
                    if self.config.saver.save_many:
                        ckpt_name = f'{self.path.save_path}/ckpt_{curr_step}.pth.tar'
                    else:
                        ckpt_name = f'{self.path.save_path}/ckpt.pth.tar'
                    self.state['model'] = self.model.state_dict()
                    self.state['optimizer'] = self.optimizer.state_dict()
                    self.state['last_iter'] = curr_step
                    torch.save(self.state, ckpt_name)

            end = time.time()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        res_file = os.path.join(self.path.result_path, f'results.txt.rank{self.dist.rank}')
        writer = open(res_file, 'w')
        for batch_idx, batch in enumerate(self.val_data['loader']):
            input = batch['image']
            label = batch['label']
            input = input.cuda()
            label = label.squeeze().view(-1).cuda().long()
            # compute output
            logits = self.model(input)
            scores = F.softmax(logits, dim=1)
            # compute prediction
            _, preds = logits.data.topk(k=1, dim=1)
            preds = preds.view(-1)
            # update batch information
            batch.update({'prediction': preds})
            batch.update({'score': scores})
            # save prediction information
            self.val_data['loader'].dataset.dump(writer, batch)

        writer.close()
        link.barrier()
        if self.dist.rank == 0:
            metrics = self.val_data['loader'].dataset.evaluate(res_file)
            self.logger.info(json.dumps(metrics.metric, indent=2))
        else:
            metrics = {}
        link.barrier()
        # broadcast metrics to other process
        metrics = broadcast_object(metrics)
        self.model.train()
        return metrics


@link_dist
def main():
    parser = argparse.ArgumentParser(description='Classification Solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()
    # build solver
    solver = ClsSolver(args.config)
    # evaluate or train
    if args.evaluate:
        if not hasattr(solver.config.saver, 'pretrain'):
            solver.logger.warn('Evaluating without resuming any solver checkpoints.')
        solver.evaluate()
    else:
        if solver.config.data.last_iter < solver.config.data.max_iter:
            solver.train()
        else:
            solver.logger.info('Training has been completed to max_iter!')


if __name__ == '__main__':
    main()
