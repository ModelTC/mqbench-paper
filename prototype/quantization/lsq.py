# type: ignore
import torch
from .base_quantizer import QuantizeBase
from .utils import pot_quantization, grad_scale
from torch.nn.parameter import Parameter


class _LearnableFakeQuantize(QuantizeBase):
    r""" This is an extension of the FakeQuantize module in fake_quantize.py, which
    supports more generalized lower-bit quantization and support learning of the scale
    and zero point parameters through backpropagation. For literature references,
    please see the class _LearnableFakeQuantizePerTensorOp.
    In addition to the attributes in the original FakeQuantize module, the _LearnableFakeQuantize
    module also includes the following attributes to support quantization parameter learning.
    """
    def __init__(self, observer, scale=1., zero_point=0., use_grad_scaling=False, **observer_kwargs):
        super(_LearnableFakeQuantize, self).__init__(observer, **observer_kwargs)
        self.use_grad_scaling = use_grad_scaling
        self.scale = Parameter(torch.tensor([scale]))
        self.zero_point = Parameter(torch.tensor([zero_point]))

    @torch.jit.export
    def toggle_qparam_learning(self, enabled=True):
        self.learning_enabled[0] = int(enabled)
        self.scale.requires_grad = enabled
        self.zero_point.requires_grad = enabled
        return self

    def extra_repr(self) -> str:
        s = 'bitwidth={bitwidth}, quant_max={quant_max}, quant_min={quant_min},' \
            'qscheme={qscheme}'
        return s.format(**self.__dict__)

    def forward(self, X):
        if self.static_enabled[0] == 1:
            if self.ada_sign and X.min() >= 0:
                self.quant_max = self.activation_post_process.quant_max = 2 ** self.bitwidth - 1
                self.quant_min = self.activation_post_process.quant_min = 0
                self.activation_post_process.adjust_sign = True
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.activation_post_process.calculate_qparams()
            _scale = _scale.to(self.scale.device)
            _zero_point = _zero_point.to(self.zero_point.device)

            if self.ch_axis != -1:
                self.scale.data = torch.ones_like(_scale)
                self.zero_point.data = torch.zeros_like(_zero_point)

            self.scale.data.copy_(_scale)
            self.zero_point.data.copy_(_zero_point)
        else:
            self.scale.data.abs_()
            self.scale.data.clamp_(min=self.eps.item())

        if self.fake_quant_enabled[0] == 1:
            if self.qscheme in (torch.per_channel_symmetric, torch.per_tensor_symmetric):
                self.zero_point.data.zero_()
            else:
                self.zero_point.data.clamp_(self.quant_min, self.quant_max)

            if self.pot_scale:
                scale = pot_quantization(self.scale)
            else:
                scale = self.scale

            if self.qscheme in (torch.per_channel_symmetric, torch.per_channel_affine):
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() / X.shape[self.ch_axis] * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                X = _fake_quantize_learnable_per_channel_affine(
                    X, scale, self.zero_point, self.ch_axis,
                    self.quant_min, self.quant_max, grad_factor)
            else:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                X = torch._fake_quantize_learnable_per_tensor_affine(
                    X, scale, self.zero_point,
                    self.quant_min, self.quant_max, grad_factor)
        return X


def _fake_quantize_learnable_per_channel_affine(x, scale, zero_point, ch_axis, quant_min, quant_max, grad_factor):
    zero_point = (zero_point.round() - zero_point).detach() + zero_point
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = grad_scale(scale, grad_factor).reshape(new_shape)
    zero_point = grad_scale(zero_point, grad_factor).reshape(new_shape)
    x = x / scale + zero_point
    x = (x.round() - x).detach() + x
    x = torch.clamp(x, quant_min, quant_max)
    return (x - zero_point) * scale
