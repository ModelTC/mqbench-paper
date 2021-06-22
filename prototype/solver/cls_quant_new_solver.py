import argparse
import time
import datetime
import torch
import copy
import spring.linklink as link
import torch.nn.functional as F
import torch.nn as nn
import math

from .cls_quant_solver import ClsSolver
from prototype.utils.dist import link_dist, DistModule, broadcast_object, dist_init
from prototype.utils.misc import makedir, create_logger, get_logger, count_params, count_flops, \
    param_group_all, AverageMeter, accuracy, load_state_model, load_state_optimizer, mixup_data, \
    mix_criterion, modify_state, cutmix_data, parse_config
from prototype.optimizer import optim_entry, FP16RMSprop, FP16SGD, FusedFP16SGD
from prototype.data import build_imagenet_train_dataloader, build_imagenet_test_dataloader
from prototype.quantization.base_quantizer import QuantizeBase
from prototype.quantization.prepare_quantization import prepare_quant_academic, enable_param_learning, \
    bitwidth_refactor, toggle_fake_quant, get_foldbn_config, get_qconfig, enable_static_observation, \
    enable_static_estimate


class ClsNewSolver(ClsSolver):

    def __init__(self, config_file):
        super(ClsNewSolver, self).__init__(config_file)

    def reset_model_bn_forward(self, model, bn_mean, bn_var):
        """
        Reset the BatchNorm forward function of the model, and store mean and variance in bn_mean and bn_var
        Args:
            model (nn.Module): the model need to be reset
            data_loader (dict): data_loader with calibrate dataset
        """
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_mean[name] = AverageMeter()
                bn_var[name] = AverageMeter()

                def new_forward(bn, mean_est, var_est):
                    def lambda_forward(x):
                        batch_mean = x.mean(0, keepdim=True).mean(
                            2, keepdim=True).mean(3,
                                                  keepdim=True)  # 1, C, 1, 1
                        batch_var = (x - batch_mean) * (x - batch_mean)
                        batch_var = batch_var.mean(0, keepdim=True).mean(
                            2, keepdim=True).mean(3, keepdim=True)

                        batch_mean = torch.squeeze(batch_mean).float()
                        batch_var = torch.squeeze(batch_var).float()
                        # 直接算正常的bn的mean 和 var

                        # 累计mean_est = batch_mean * batch
                        reduce_batch_mean = batch_mean.clone(
                        ) / link.get_world_size()
                        reduce_batch_var = batch_var.clone(
                        ) / link.get_world_size()
                        link.allreduce(reduce_batch_mean.data)
                        link.allreduce(reduce_batch_var.data)
                        mean_est.update(reduce_batch_mean.data, x.size(0))
                        var_est.update(reduce_batch_var.data, x.size(0))

                        # bn forward using calculated mean & var
                        _feature_dim = batch_mean.size(0)
                        return F.batch_norm(
                            x,
                            batch_mean,
                            batch_var,
                            bn.weight[:_feature_dim],
                            bn.bias[:_feature_dim],
                            False,
                            0.0,
                            bn.eps,
                        )

                    return lambda_forward

                m.forward = new_forward(m, bn_mean[name], bn_var[name])

    def reset_subnet_running_statistics(self, model, data_loader):
        """
        Recalculate model BatchNorm mean and variance with the calibrate data_loader
        Args:
            model (nn.Module): the model need to be reset
            data_loader (dict): data_loader with calibrate dataset
        """
        bn_mean = {}
        bn_var = {}
        forward_model = copy.deepcopy(model.module)
        forward_model.cuda()
        forward_model = DistModule(forward_model, True)
        self.reset_model_bn_forward(forward_model, bn_mean, bn_var)
        self.logger.info('calculate bn')
        with torch.no_grad():
            for batch in data_loader:
                images = batch['image']
                images = images.cuda()
                forward_model(images)

        for name, m in model.named_modules():
            if name in bn_mean and bn_mean[name].count > 0:
                feature_dim = bn_mean[name].avg.size(0)
                assert isinstance(m, nn.BatchNorm2d)
                # 最后取得的均值
                m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
                m.running_var.data[:feature_dim].copy_(bn_var[name].avg)
        self.logger.info('bn complete')
        return model

    def build_calib_dataset(self, image_size=224):
        self.logger.info('build calib dataset')
        config = copy.deepcopy(self.config.data)
        config.input_size = image_size
        config.test_resize = math.ceil(image_size / 0.875)
        config.train.meta_file = '/mnt/lustre/share/shenmingzhu/ImageNet/train_4k.txt'
        config.max_iter = 4096 // self.dist.world_size // config.batch_size
        config.last_iter = 0
        self.calib_data = build_imagenet_train_dataloader(config)

    def calib_bn(self, model):
        model = self.reset_subnet_running_statistics(model, self.calib_data['loader'])
        return model

    def initialize_scale_and_zero_point(self):
        self.quant_scale = {}
        self.quant_zero_point = {}
        for name, module in self.model.named_modules():
            if isinstance(module, QuantizeBase):
                self.quant_scale[name] = AverageMeter()
                self.quant_zero_point[name] = AverageMeter()

    def accumalate_scale_and_zero_point(self):
        for name, module in self.model.named_modules():
            if isinstance(module, QuantizeBase):
                self.quant_scale[name].update(module.scale.data)
                self.quant_zero_point[name].update(module.zero_point.data)
                self.logger.info('layer {} scale {} zero_point {}'.format(
                    name, module.scale.data, module.zero_point.data))

    def reset_scale_and_zero_point(self):
        for name, module in self.model.named_modules():
            if isinstance(module, QuantizeBase):
                module.scale.data = torch.tensor([self.quant_scale[name].avg])
                module.zero_point.data = torch.tensor([self.quant_zero_point[name].avg])
                self.logger.info('after reset layer {} scale {} zero_point {}'.format(
                    name, module.scale.data, module.zero_point.data))

    def print_scale_and_zero_point(self):
        for name, module in self.model.named_modules():
            if isinstance(module, QuantizeBase):
                self.logger.info('after reset layer {} scale {} zero_point {}'.format(
                    name, module.scale.data, module.zero_point.data))

    def print_min_max_val(self):
        for name, module in self.model.named_modules():
            if isinstance(module, QuantizeBase):
                self.logger.info('after reset layer {} min_val {} max_val {}'.format(
                    name, module.activation_post_process.min_val.data,
                    module.activation_post_process.max_val.data))

    def initialize(self, i, curr_step, flag=True, calib_bn=True):
        if i != 0 and curr_step != 1:
            return
        if i == 0 and curr_step != 1:
            enable_param_learning(self.model)
            return
        # forward to initialize scale and zero point
        # reset statistics
        self.build_calib_dataset()

        #self.initialize_scale_and_zero_point()
        self.model.eval()
        if flag:
            self.logger.info('enable_static_observation')
            enable_static_observation(self.model)
        else:
            self.logger.info('enable_static_estimate')
            enable_static_estimate(self.model)
        if i == 0 and curr_step == 1:
            for batch in self.calib_data['loader']:
                images = batch['image']
                images = images.cuda()
                self.model(images)

        # change the fp32 weight and input into quantization
        # in the forward pass, the quant input and quant weight are used
        enable_param_learning(self.model)

        # reset running statistics and evaluate the quant model to verify the benefits of initialization
        if i == 0 and curr_step == 1:
            if calib_bn:
                self.build_calib_dataset()
                self.reset_subnet_running_statistics(self.model, self.calib_data['loader'])
            self.logger.info('evaluate the impact of initialization')
            self.evaluate()

        if self.dist.rank == 0:
            name = self.config.qparams.a_observer + '_' + self.config.qparams.w_observer
            ckpt_name = f'{self.path.save_path}/{name}_ckpt_init.pth.tar'
            self.state['model'] = self.model.state_dict()
            torch.save(self.state, ckpt_name)
        return

    def train(self):

        self.pre_train()
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
            input = input.cuda().half() if self.fp16 else input.cuda()
            # mixup
            if self.mixup < 1.0:
                input, target_a, target_b, lam = mixup_data(input, target, self.mixup)
            # cutmix
            if self.cutmix > 0.0:
                input, target_a, target_b, lam = cutmix_data(input, target, self.cutmix)
            # forward
            self.initialize(i, curr_step)

            logits = self.model(input)

            # mixup
            if self.mixup < 1.0 or self.cutmix > 0.0:
                loss = mix_criterion(self.criterion, logits, target_a, target_b, lam)
                loss /= self.dist.world_size
            else:
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
            if FusedFP16SGD is not None and isinstance(self.optimizer, FusedFP16SGD):
                self.optimizer.backward(loss)
                # self.model.sync_gradients()
                self.optimizer.step()
            elif isinstance(self.optimizer, FP16SGD) or isinstance(self.optimizer, FP16RMSprop):

                def closure():
                    self.optimizer.backward(loss, False)
                    self.model.sync_gradients()
                    # check overflow, convert to fp32 grads, downscale
                    self.optimizer.update_master_grads()
                    return loss
                self.optimizer.step(closure)
            else:
                loss.backward()
                self.model.sync_gradients()
                self.optimizer.step()
            # EMA
            if self.ema is not None:
                self.ema.step(self.model, curr_step=curr_step)
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
                if self.ema is not None:
                    self.ema.load_ema(self.model)
                    ema_metrics = self.evaluate()
                    self.ema.recover(self.model)
                    if self.dist.rank == 0 and self.config.data.test.evaluator.type == 'imagenet':
                        metric_key = 'top{}'.format(self.topk)
                        self.tb_logger.add_scalars('acc1_val', {'ema': ema_metrics.metric['top1']}, curr_step)
                        self.tb_logger.add_scalars('acc5_val', {'ema': ema_metrics.metric[metric_key]}, curr_step)

                # testing logger
                if self.dist.rank == 0 and self.config.data.test.evaluator.type == 'imagenet':
                    metric_key = 'top{}'.format(self.topk)
                    self.tb_logger.add_scalar('acc1_val', metrics.metric['top1'], curr_step)
                    self.tb_logger.add_scalar('acc5_val', metrics.metric[metric_key], curr_step)

                # save ckpt
                if self.dist.rank == 0:
                    if self.config.saver.save_many:
                        ckpt_name = f'{self.path.save_path}/ckpt_{curr_step}.pth.tar'
                    else:
                        ckpt_name = f'{self.path.save_path}/ckpt.pth.tar'
                    self.state['model'] = self.model.state_dict()
                    self.state['optimizer'] = self.optimizer.state_dict()
                    self.state['last_iter'] = curr_step
                    if self.ema is not None:
                        self.state['ema'] = self.ema.state_dict()
                    torch.save(self.state, ckpt_name)

            end = time.time()


@link_dist
def main():
    parser = argparse.ArgumentParser(description='Classification Solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()
    # build solver
    solver = ClsNewSolver(args.config)
    # evaluate or train
    if args.evaluate:
        if not hasattr(solver.config.saver, 'pretrain'):
            solver.logger.warn('Evaluating without resuming any solver checkpoints.')
        solver.evaluate()
        if solver.ema is not None:
            solver.ema.load_ema(solver.model)
            solver.evaluate()
    else:
        if solver.config.data.last_iter < solver.config.data.max_iter:
            solver.train()
        else:
            solver.logger.info('Training has been completed to max_iter!')


if __name__ == '__main__':
    main()
