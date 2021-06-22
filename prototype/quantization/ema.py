import torch
from torch.nn import Module
from torch.quantization.observer import MovingAverageMinMaxObserver
from prototype.quantization.lsq import _LearnableFakeQuantize
import spring.linklink as link


class FixedFakeQuantize(_LearnableFakeQuantize):

    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(self, observer=MovingAverageMinMaxObserver, quant_min=-128, quant_max=127, **observer_kwargs):
        super().__init__(observer=observer, quant_min=quant_min, quant_max=quant_max, **observer_kwargs)

    def forward(self, X):

        if self.static_enabled[0] == 1:
            self.activation_post_process(X.detach())
            if self.ada_sign and X.min() >= 0:
                self.quant_max = 2 ** self.bitwidth - 1
                self.quant_min = 0

            if self.lsq_init is True:
                max_val, min_val = self.activation_post_process.max_val, self.activation_post_process.min_val
                _scale = (max_val - min_val) / (2 ** self.bitwidth - 1)
                _zero_point = torch.zeros_like(_scale)
            else:
                _scale, _zero_point = self.activation_post_process.calculate_qparams()
                _scale = _scale.to(self.scale.device)
                _zero_point = _zero_point.to(self.zero_point.device)

            link.allreduce(_scale)
            _scale = _scale / link.get_world_size()
            self.scale.data.copy_(_scale)
            self.zero_point.data.copy_(_zero_point)
        else:
            self.scale.data.abs_()
            self.scale.data.clamp_(min=self.eps.item())

        if self.fake_quant_enabled[0] == 1:
            if self.qscheme in (torch.per_channel_symmetric, torch.per_tensor_symmetric):
                self.zero_point.data.zero_()

            grad_factor = 0.

            if self.qscheme in (
                    torch.per_channel_symmetric, torch.per_channel_affine):
                X = torch._fake_quantize_learnable_per_channel_affine(
                    X, self.scale, self.zero_point, self.ch_axis,
                    self.quant_min, self.quant_max, grad_factor)
            else:
                X = torch._fake_quantize_learnable_per_tensor_affine(
                    X, self.scale, self.zero_point,
                    self.quant_min, self.quant_max, grad_factor)
        return X