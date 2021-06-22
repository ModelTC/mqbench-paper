from .cls_quant_new_solver import ClsNewSolver
import os
import argparse
from easydict import EasyDict
from tensorboardX import SummaryWriter
import pprint
import torch
import spring.linklink as link
from prototype.utils.dist import link_dist, DistModule, broadcast_object, dist_init
from prototype.utils.misc import makedir, create_logger, get_logger, count_params, count_flops, \
    param_group_all, AverageMeter, accuracy, load_state_model, load_state_optimizer, mixup_data, \
    mix_criterion, modify_state, cutmix_data, parse_config

class ClsEvalSolver(ClsNewSolver):

    def __init__(self, config_file, a_observer, w_observer):
        self.a_observer = a_observer
        self.w_observer = w_observer

        super(ClsEvalSolver, self).__init__(config_file)
        self.logger.info('current observer a_observer {} w_observer {}'.format(self.a_observer, self.w_observer))

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
        name = self.a_observer + '_' + self.w_observer + '_'
        create_logger(os.path.join(self.path.root_path, name+'log.txt'))
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
        torch.backends.cudnn.benchmark = True

    def train(self,):
        self.config.qparams.a_observer = self.a_observer
        self.config.qparams.w_observer = self.w_observer
        self.build_model()
        self.initialize(0, 1, flag=True, calib_bn=True)


@link_dist
def main():
    parser = argparse.ArgumentParser(description='Classification Solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--a_observer', default='MinMaxObserver', type=str)
    parser.add_argument('--w_observer', default='MinMaxObserver', type=str)

    args = parser.parse_args()
    solver = ClsEvalSolver(args.config, args.a_observer, args.w_observer)
    solver.build_model()
    solver.train()


if __name__ == '__main__':
    main()
