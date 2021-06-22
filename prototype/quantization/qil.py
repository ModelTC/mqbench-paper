# type: ignore
import torch
from .base_quantizer import QuantizeBase
from .utils import pot_quantization, grad_scale, sync_tensor
from torch.nn.parameter import Parameter


class _QILQuantize(QuantizeBase):

    def __init__(self, observer, quantize_weight=True, **observer_kwargs):
        super(_QILQuantize, self).__init__(observer, **observer_kwargs)
        self.quantize_weight = quantize_weight
        self.center = Parameter(torch.tensor([float('inf')]))
        self.distance = Parameter(torch.tensor([float('inf')]))
        if quantize_weight:
            self.gamma = Parameter(torch.tensor(1.0))
        self.sign = True

    @torch.jit.export
    def toggle_qparam_learning(self, enabled=True):
        self.learning_enabled[0] = int(enabled)
        self.center.requires_grad = enabled
        self.distance.requires_grad = enabled

        # self.gamma.requires_grad = enabled
        return self

    def extra_repr(self) -> str:
        s = 'bitwidth={bitwidth}, quant_max={quant_max}, quant_min={quant_min},' \
            'qscheme={qscheme}'
        return s.format(**self.__dict__)

    def forward(self, X: torch.Tensor):
        if self.static_enabled[0] == 1:
            if self.ada_sign and X.min() >= 0:
                self.sign = False
                self.activation_post_process.adjust_sign = True

            self.activation_post_process(X.detach())

        if self.fake_quant_enabled[0] == 1:
            if self.center.data == float('inf'):
                self.center.data = X.abs().max() * 0.5
                self.distance.data = X.abs().max() * 0.5
                if not self.quantize_weight:
                    sync_tensor(self.center.data)
                    sync_tensor(self.distance.data)

            self.center.data.abs_()
            self.distance.data.abs_()
            self.center.data.clamp_(max=(X.abs().max() * 0.5).item())
            self.distance.data.clamp_(max=(X.abs().max() * 0.5).item())

            if self.bitwidth == 8:
                _scale, _zero_point = self.activation_post_process.calculate_qparams()
                X = torch.fake_quantize_per_tensor_affine(
                    X, float(_scale), int(_zero_point), self.quant_min, self.quant_max)
            else:
                grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
                center = grad_scale(self.center, grad_factor)
                distance = grad_scale(self.distance, grad_factor)

                alpha = 0.5 / distance
                beta = -0.5 * (center / distance) + 0.5

                if self.quantize_weight:
                    # gamma = grad_scale(self.gamma, grad_factor)
                    X = torch.clamp((X.abs() * alpha + beta), 0, 1) * X.sign()
                    X = symmetric_quantization(X, delta=2 ** (self.bitwidth-1) - 1)

                else:
                    if self.sign:
                        X = torch.clamp((X.abs() * alpha + beta), 0, 1) * X.sign()
                        X = symmetric_quantization(X, delta=2 ** (self.bitwidth - 1) - 1)
                    else:
                        X = X * alpha + beta
                        X = symmetric_quantization(X, delta=2 ** self.bitwidth - 1)

        return X


def symmetric_quantization(X, delta):
    X = X * delta
    X = (X.round() - X).detach() + X
    return X / delta
