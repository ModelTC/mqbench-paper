import torch
import math
from torch.quantization.observer import _ObserverBase
from prototype.quantization.utils import is_symmetric, sync_tensor
from scipy.optimize import minimize_scalar
import numpy as np
import warnings
from prototype.utils.misc import get_logger


class ObserverBase(_ObserverBase):

    def __init__(self, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ada_sign=True,
                 pot_scale=False, reduce_range=False, quant_min=-128, quant_max=127, ch_axis=-1):
        super(ObserverBase, self).__init__(dtype=dtype, qscheme=qscheme, reduce_range=reduce_range,
                                           quant_min=quant_min, quant_max=quant_max)
        self.ch_axis = ch_axis
        self.ada_sign = ada_sign
        self.pot_scale = pot_scale
        self.adjust_sign = False
        self.register_buffer('min_val', torch.tensor(float('inf')))
        self.register_buffer('max_val', torch.tensor(float('-inf')))

    def _calculate_qmin_qmax(self):
        return self.quant_min, self.quant_max

    def calculate_qparams(self):
        return self._calculate_qparams(self.min_val, self.max_val)

    def _calculate_qparams(self, min_val, max_val):
        r"""Calculates the quantization parameters, given min and max
        value tensors. Works for both per tensor and per channel cases
        Args:
            min_val: Minimum values per channel
            max_val: Maximum values per channel
        Returns:
            scales: Scales tensor of shape (#channels,)
            zero_points: Zero points tensor of shape (#channels,)
        """
        if min_val.numel() == 0 or max_val.numel() == 0:
            warnings.warn(
                "must run observer before calling calculate_qparams.\
                                    Returning default scale and zero point "
            )
            return torch.tensor([1.0]), torch.tensor([0])

        if min_val.dim() == 0 or max_val.dim() == 0:
            if min_val == float('inf') and max_val == float('-inf'):
                warnings.warn(
                    "must run observer before calling calculate_qparams.\
                                        Returning default scale and zero point "
                )
                return torch.tensor([1.0]), torch.tensor([0])

            assert min_val <= max_val, "min {} should be less than max {}".format(
                min_val, max_val
            )
        else:
            assert torch.all(min_val <= max_val), "min {} should be less than max {}".format(
                min_val, max_val
            )

        quant_min, quant_max = self._calculate_qmin_qmax()
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

        if self.qscheme == torch.per_tensor_symmetric or self.qscheme == torch.per_channel_symmetric:
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            if self.adjust_sign:
                scale = max_val_pos / (float(quant_max - quant_min))
            else:
                scale = max_val_pos / (float(quant_max - quant_min) / 2)
            scale = torch.max(scale, self.eps)
            if self.dtype == torch.quint8:
                if self.has_customized_qrange:
                    # When customized quantization range is used, down-rounded midpoint of the range is chosen.
                    zero_point = zero_point.new_full(zero_point.size(), (quant_min + quant_max) // 2)
                else:
                    zero_point = zero_point.new_full(zero_point.size(), 128)
        elif self.qscheme == torch.per_channel_affine_float_qparams:
            scale = (max_val - min_val) / float(quant_max - quant_min)
            scale = torch.where(scale > self.eps, scale, torch.ones_like(scale))
            # We use the quantize function
            # xq = Round(Xf * inv_scale + zero_point),
            # setting zero_point to (-1 * min *inv_scale) we get
            # Xq = Round((Xf - min) * inv_scale)
            zero_point = -1 * min_val / scale
        else:
            scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
            scale = torch.max(scale, self.eps)
            zero_point = quant_min - torch.round(min_val_neg / scale)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)

        # For scalar values, cast them to Tensors of size 1 to keep the shape
        # consistent with default values in FakeQuantize.
        if len(scale.shape) == 0:
            # TODO: switch to scale.item() after adding JIT support
            scale = torch.tensor([float(scale)], dtype=scale.dtype, device=device)
        if len(zero_point.shape) == 0:
            # TODO: switch to zero_point.item() after adding JIT support
            zero_point = torch.tensor([int(zero_point)], dtype=zero_point.dtype, device=device)
            if self.qscheme == torch.per_channel_affine_float_qparams:
                zero_point = torch.tensor([float(zero_point)], dtype=zero_point.dtype, device=device)

        return scale, zero_point

    def extra_repr(self) -> str:
        s = 'qscheme={}, min_val={}, max_val={}'.format(self.qscheme, self.min_val.mean().item(),
                                                        self.max_val.mean().item())
        return s


# Always use the current min and max of all batch to calculate qparams
class MinMaxObserver(ObserverBase):
    def __init__(self, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ada_sign=True,
                 pot_scale=False, reduce_range=False, quant_min=None, quant_max=None, ch_axis=-1):

        super(MinMaxObserver, self).__init__(dtype, qscheme, ada_sign, pot_scale, reduce_range, quant_min, quant_max,
                                             ch_axis)

    def forward(self, x):
        r"""Records the running minimum and maximum of ``x``."""
        if x.numel() == 0:
            return x
        # if self.ada_sign and x.min() >= 0:
        #     self.quant_max = self.quant_max - self.quant_min
        #     self.quant_min = 0
        x = x.to(self.min_val.dtype)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
        else:
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            min_val_cur, max_val_cur = torch._aminmax(y, 1)

        # self.min_val = min(min_val_cur, self.min_val)
        # self.max_val = max(max_val_cur, self.max_val)
        self.min_val = min_val_cur
        self.max_val = max_val_cur

        return x


class AverageMinMaxObserver(ObserverBase):
    def __init__(self, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ada_sign=True,
                 pot_scale=False, reduce_range=False, quant_min=None, quant_max=None, ch_axis=-1):

        super(AverageMinMaxObserver, self).__init__(dtype, qscheme, ada_sign, pot_scale, reduce_range, quant_min, quant_max,
                                             ch_axis)
        self.iter = 0
        self.min_val = None
        self.max_val = None

    def forward(self, x):
        r"""Records the running minimum and maximum of ``x``."""
        if x.numel() == 0:
            return x
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
        else:
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            min_val_cur, max_val_cur = torch._aminmax(y, 1)

        self.iter += 1
        if not self.min_val and not self.max_val:
            self.min_val = min_val_cur
            self.max_val = max_val_cur
        else:
            self.min_val = (self.min_val * self.iter + min_val_cur) / (self.iter + 1)
            self.max_val = (self.max_val * self.iter + max_val_cur) / (self.iter + 1)
        return x


class ClipStdObserver(ObserverBase):

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine, std_scale=2.6, ada_sign=True,
                 pot_scale=False, reduce_range=False, quant_min=None, quant_max=None, ch_axis=-1):

        super(ClipStdObserver, self).__init__(dtype, qscheme, ada_sign, pot_scale, reduce_range, quant_min, quant_max,
                                              ch_axis)
        self.std_scale = std_scale

    def forward(self, x):
        r"""Records the running minimum and maximum of ``x``."""
        if x.numel() == 0:
            return x
        x = x.to(self.min_val.dtype)
        if self.ch_axis == -1:
            mean = x.mean()
            std = x.std()
        else:
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            mean = y.mean(1)
            std = y.std(1)

        # using statistics to clip min and max
        min_val = mean - self.std_scale * std
        max_val = mean + self.std_scale * std

        self.min_val = min_val
        self.max_val = max_val

        return x


class BatchMeanMaxObserver(ObserverBase):
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine, momentum=0.9, ada_sign=True,
                 pot_scale=False, reduce_range=False, quant_min=None, quant_max=None, ch_axis=-1):

        super(BatchMeanMaxObserver, self).__init__(dtype, qscheme, ada_sign, pot_scale, reduce_range, quant_min,
                                                   quant_max, ch_axis)
        self.momentum = momentum

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.view(x_orig.size(0), -1)
        min_val = x.min(-1)[0].mean(-1)
        max_val = x.max(-1)[0].mean(-1)

        sync_tensor(max_val)
        sync_tensor(min_val)

        if self.min_val == float('inf'):
            self.min_val = min_val
            self.max_val = max_val
        else:
            self.min_val = self.min_val * self.momentum + min_val * (1 - self.momentum)
            self.max_val = self.max_val * self.momentum + max_val * (1 - self.momentum)

        return x_orig


class QuantileMovingAverageObserver(ObserverBase):

    def __init__(self, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ada_sign=True,
                 pot_scale=False, reduce_range=False, quant_min=-128, quant_max=128, ch_axis=-1,
                 averaging_constant=0.01, quantile=0.9999):
        super(QuantileMovingAverageObserver, self).__init__(dtype, qscheme, ada_sign, pot_scale, reduce_range,
                                                            quant_min, quant_max, ch_axis)
        self.averaging_constant = averaging_constant
        self.quantile = quantile

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val = self.min_val
        max_val = self.max_val
        if min_val == float('inf') and max_val == float('-inf'):
            min_val, max_val = torch.quantile(x, 1-self.quantile), torch.quantile(x, self.quantile)
            self.min_val.resize_(min_val.shape)
            self.max_val.resize_(max_val.shape)
            self.min_val.copy_(min_val)
            self.max_val.copy_(max_val)

        return x_orig


class LSQObserver(ObserverBase):
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(self, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ada_sign=True,
                 pot_scale=False, reduce_range=False, quant_min=-128, quant_max=128, ch_axis=-1):

        super(LSQObserver, self).__init__(dtype, qscheme, ada_sign, pot_scale, reduce_range, quant_min, quant_max,
                                          ch_axis)
        self.ch_axis = ch_axis
        self.ada_sign = ada_sign
        self.tensor_norm = None

    def forward(self, x):
        if x.numel() == 0:
            return x
        if self.ch_axis == -1:
            self.tensor_norm = x.abs().mean()
            self.min_val, self.max_val = torch._aminmax(x)
        else:
            # compute channel-wise mean
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            self.tensor_norm = y.abs().mean(1)
            self.min_val, self.max_val = torch._aminmax(y)

        return x

    def calculate_qparams(self,):

        sync_tensor(self.tensor_norm)
        scale = 2 * self.tensor_norm / math.sqrt(self.quant_max)
        zero_point = torch.zeros_like(self.tensor_norm)
        if self.qscheme in [torch.per_channel_affine, torch.per_tensor_affine]:
            if self.min_val >= 0.:
                zero_point = self.quant_min - torch.round(self.min_val / scale)
        return scale, zero_point


class AverageLSQObserver(ObserverBase):
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(self, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ada_sign=True,
                 pot_scale=False, reduce_range=False, quant_min=-128, quant_max=128, ch_axis=-1):

        super(AverageLSQObserver, self).__init__(dtype, qscheme, ada_sign, pot_scale, reduce_range, quant_min, quant_max,
                                          ch_axis)
        self.ch_axis = ch_axis
        self.ada_sign = ada_sign
        self.tensor_norm = None
        self.iter = 0

    def forward(self, x):
        if x.numel() == 0:
            return x
        if self.ch_axis == -1:
            if not self.tensor_norm:
                self.tensor_norm = x.abs().mean()
            else:
                self.tensor_norm = (self.tensor_norm * self.iter + x.abs().mean()) / (self.iter + 1)
            self.iter += 1
            self.min_val, self.max_val = torch._aminmax(x)
        else:
            # compute channel-wise mean
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            self.tensor_norm = y.abs().mean(1)
            self.min_val, self.max_val = torch._aminmax(y)

        return x

    def calculate_qparams(self, sync=True):

        scale = 2 * self.tensor_norm / math.sqrt(self.quant_max)
        zero_point = torch.zeros_like(self.tensor_norm)
        if self.qscheme in [torch.per_channel_affine, torch.per_tensor_affine]:
            if self.min_val >= 0.:
                zero_point = self.quant_min - torch.round(self.min_val / scale)
        if sync:
            sync_tensor(scale)
            sync_tensor(zero_point)
        return scale, zero_point

    def extra_repr(self):
        mean_max, mean_min, mean_norm = self.min_val.mean(), self.max_val.mean(), self.tensor_norm.mean()
        s = 'max_val={:3f}, min_val={:3f}, tensor_norm={:3f}'.format(mean_max, mean_min, mean_norm)
        return s


class LSQPlusObserver(ObserverBase):
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(self, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ada_sign=True,
                 pot_scale=False, reduce_range=False, quant_min=-128, quant_max=128, ch_axis=-1):

        super(LSQPlusObserver, self).__init__(dtype, qscheme, ada_sign, pot_scale, reduce_range, quant_min, quant_max,
                                          ch_axis)
        self.ch_axis = ch_axis
        self.ada_sign = ada_sign
        self.mean = None
        self.std = None

    def forward(self, x):
        if x.numel() == 0:
            return x
        if self.ch_axis == -1:
            self.mean = x.mean()
            self.std = x.std()
            self.min_val, self.max_val = torch._aminmax(x)
        else:
            # compute channel-wise mean
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            self.mean = y.mean(1)
            self.std = y.std(1)
            self.min_val, self.max_val = torch._aminmax(y)

        return x

    def calculate_qparams(self, sync=True):

        scale = torch.maximum((self.mean-3*self.std).abs(),
                              (self.mean+3*self.std).abs()) / ((self.quant_max-self.quant_min)//2)
        zero_point = torch.zeros_like(self.mean)
        if self.qscheme in [torch.per_channel_affine, torch.per_tensor_affine]:
            if self.min_val >= 0.:
                zero_point = self.quant_min - torch.round(self.min_val / scale)
        if sync:
            sync_tensor(scale)
            sync_tensor(zero_point)
        return scale, zero_point

    def extra_repr(self):
        mean_max, mean_min, mean_norm = self.min_val.mean(), self.max_val.mean(), self.tensor_norm.mean()
        s = 'max_val={:3f}, min_val={:3f}, tensor_norm={:3f}'.format(mean_max, mean_min, mean_norm)
        return s


class MSEObserver(MinMaxObserver):
    def __init__(self, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ada_sign=True,
                 pot_scale=False, reduce_range=False, quant_min=None, quant_max=None, ch_axis=-1):

        super(MSEObserver, self).__init__(dtype, qscheme, ada_sign, pot_scale, reduce_range, quant_min, quant_max,
                                             ch_axis)

    def forward(self, x):
        r"""Records the running minimum and maximum of ``x``."""
        super().forward(x)
        logger = get_logger(__name__)
        logger.info('before calculate min val {} max val {}'.format(self.min_val, self.max_val))

        def mse_func(max_val):
            self.max_val = torch.from_numpy(np.array(max_val))
            _scale, _zero_point = self.calculate_qparams()
            if self.qscheme in (torch.per_channel_symmetric, torch.per_channel_affine):
                quant_x = torch.fake_quantize_per_channel_affine(
                    x, _scale, _zero_point.long(), self.ch_axis, self.quant_min, self.quant_max)
            else:
                quant_x = torch.fake_quantize_per_tensor_affine(
                    x, float(_scale), int(_zero_point), self.quant_min, self.quant_max)
            return (quant_x - x).pow(2).sum().cpu().numpy()

        val = np.array(self.max_val.cpu().numpy())
        res = minimize_scalar(mse_func, np.array(val), method='Bounded', bounds=[0.3*val, 1.0*val])
        self.max_val = (torch.from_numpy(np.array(res.x))).to(x.dtype)
        logger.info('after calculate min val {} max val {}'.format(self.min_val, self.max_val))

        return x


class KLDObserver(MinMaxObserver):
    def __init__(self, dtype=torch.qint8, qscheme=torch.per_tensor_affine, ada_sign=True,
                 pot_scale=False, reduce_range=False, quant_min=None, quant_max=None, ch_axis=-1):

        super(KLDObserver, self).__init__(dtype, qscheme, ada_sign, pot_scale, reduce_range, quant_min, quant_max,
                                             ch_axis)
        self.observer = MinMaxObserver(dtype=dtype, qscheme=qscheme, ada_sign=ada_sign,
                                       pot_scale=pot_scale, reduce_range=reduce_range,
                                       quant_min=quant_min, quant_max=quant_max, ch_axis=ch_axis)

    def forward(self, x):
        r"""Records the running minimum and maximum of ``x``."""
        super().forward(x)
        logger = get_logger(__name__)
        logger.info('before calculate min val {} max val {}'.format(self.min_val, self.max_val))

        def mse_func(max_val):
            self.max_val = torch.from_numpy(np.array(max_val))
            _scale, _zero_point = self.calculate_qparams()
            if self.qscheme in (torch.per_channel_symmetric, torch.per_channel_affine):
                quant_x = torch.fake_quantize_per_channel_affine(
                    x, _scale, _zero_point.long(), self.ch_axis, self.quant_min, self.quant_max)
            else:
                quant_x = torch.fake_quantize_per_tensor_affine(
                    x, float(_scale), int(_zero_point), self.quant_min, self.quant_max)
            hist_quant_x = torch.histc(quant_x, bins=2048)
            hist_quant_x = torch.max(hist_quant_x, torch.zeros_like(hist_quant_x).fill_(1e-9))
            hist_x = torch.histc(x, bins=2048)
            kld = (hist_x * torch.log(hist_x / hist_quant_x)).sum().cpu().numpy()
            return kld

        val = np.array(self.max_val.cpu().numpy())
        res = minimize_scalar(mse_func, np.array(val), method='Bounded', bounds=[0.3*val, 1.0*val])
        self.max_val = (torch.from_numpy(np.array(res.x))).to(x.dtype)
        logger.info('after calculate min val {} max val {}'.format(self.min_val, self.max_val))

        return x


if __name__ == '__main__':
    observer = KLDObserver(quant_min=-8, quant_max=7, qscheme=torch.per_tensor_symmetric)
    x = torch.randn(1, 3, 4, 4)
    print(x)
    observer(x)
    max_val_pos = torch.max(-observer.observer.min_val, observer.observer.max_val)
    scale = max_val_pos / (float(observer.observer.quant_max - observer.observer.quant_min) / 2)
    print('calculate', observer.observer.quant_max, observer.observer.quant_min, scale)
    scale, zero_point = observer.calculate_qparams()
    print(scale, zero_point)
    x = torch.randn(1, 3, 4, 4)
    print(x)
    observer(x)
    scale, zero_point = observer.calculate_qparams()
    max_val_pos = torch.max(-observer.observer.min_val, observer.observer.max_val)
    scale = max_val_pos / (float(observer.observer.quant_max - observer.observer.quant_min) / 2)
    print('calculate', scale)
    print(scale, zero_point)
