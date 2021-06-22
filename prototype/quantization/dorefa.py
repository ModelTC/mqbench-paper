# type: ignore
import torch
from .base_quantizer import QuantizeBase
from .utils import is_symmetric, is_per_channel, pot_quantization


class _DoReFaFakeQuantize(QuantizeBase):
    def __init__(self, observer, quantize_weight=True, **observer_kwargs):
        super(_DoReFaFakeQuantize, self).__init__(observer, **observer_kwargs)
        self.quantize_weight = quantize_weight

    def extra_repr(self) -> str:
        s = 'bitwidth={bitwidth}, quant_max={quant_max}, quant_min={quant_min}, quantize_weight={quantize_weight}, ' \
            'qscheme={qscheme}, ada_sign={ada_sign}'
        return s.format(**self.__dict__)

    def forward(self, X):
        if 1:
            if self.quantize_weight:
                X = torch.tanh(X)
                X = X.div(X.abs().max() + 1e-5)
                self.activation_post_process(X.detach())
            else:
                if self.ada_sign and X.min() >= 0:
                    self.quant_max = self.activation_post_process.quant_max = 2 ** self.bitwidth - 1
                    self.quant_min = self.activation_post_process.quant_min = 0
                    self.activation_post_process.adjust_sign = True
                if self.bitwidth == 8:
                    self.activation_post_process(X.detach())
                else:
                    self.activation_post_process(torch.clamp(X.detach(), -1., 1.))
                    self.activation_post_process.max_val.data.fill_(1.)
                    if X.min() < 0:
                        self.activation_post_process.min_val.data.fill_(-1.)
                    else:
                        self.activation_post_process.min_val.data.fill_(0.)

            _scale, _zero_point = self.activation_post_process.calculate_qparams()

            # self.scale = _scale
            # self.zero_point = _zero_point

        if self.fake_quant_enabled[0] == 1:
            if self.pot_scale:
                _scale = pot_quantization(_scale)
            if is_symmetric(self.qscheme):
                _zero_point.data.zero_()

            if is_per_channel(self.qscheme):
                X = torch.fake_quantize_per_channel_affine(
                    X, _scale, _zero_point.long(), self.ch_axis, self.quant_min, self.quant_max)
            else:
                X = torch.fake_quantize_per_tensor_affine(
                    X, float(_scale), int(_zero_point), self.quant_min, self.quant_max)
        return X
