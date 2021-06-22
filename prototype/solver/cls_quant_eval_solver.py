from .cls_quant_new_solver import ClsNewSolver
from prototype.utils.dist import link_dist
import argparse


class ClsEvalSolver(ClsNewSolver):

    def __init__(self, config_file):
        super(ClsEvalSolver, self).__init__(config_file)

    def train(self):
        self.initialize(0, 1, flag=True, calib_bn=True)


@link_dist
def main():
    parser = argparse.ArgumentParser(description='Classification Solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()
    # build solver
    solver = ClsEvalSolver(args.config)
    # evaluate or train
    solver.train()



if __name__ == '__main__':
    main()
