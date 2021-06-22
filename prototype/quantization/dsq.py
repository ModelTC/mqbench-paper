# type: ignore
import torch
import math
from .base_quantizer import QuantizeBase
from .utils import pot_quantization, is_per_channel


def dsq_function_per_tensor(x, scale, zero_point, quant_min, quant_max, alpha):
    tanh_scale = 1 / (1 - alpha)
    tanh_k = math.log((tanh_scale + 1) / (tanh_scale - 1))

    x = x / scale + zero_point
    x = torch.clamp(x, quant_min, quant_max)
    x_f = x.floor()
    x = x_f + (tanh_scale * torch.tanh(tanh_k * (x-x_f-0.5))) * 0.5 + 0.5
    x = (x.round() - x).detach() + x
    x = (x - zero_point) * scale

    return x


def dsq_function_per_channel(x, scale, zero_point, quant_min, quant_max, ch_axis, alpha):

    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = scale.reshape(new_shape)
    zero_point = zero_point.reshape(new_shape)

    tanh_scale = 1 / (1 - alpha)
    tanh_k = math.log((tanh_scale + 1) / (tanh_scale - 1))

    x = x / scale + zero_point
    x = torch.clamp(x, quant_min, quant_max)
    x_f = x.floor()
    x = x_f + (tanh_scale * torch.tanh(tanh_k * (x - x_f - 0.5))) * 0.5 + 0.5
    x = (x.round() - x).detach() + x
    x = (x - zero_point) * scale

    return x


class _DSQFakeQuantize(QuantizeBase):
    def __init__(self, observer, alpha=0.4, **observer_kwargs):
        super(_DSQFakeQuantize, self).__init__(observer, **observer_kwargs)
        self.alpha = alpha

    def extra_repr(self) -> str:
        s = 'bitwidth={bitwidth}, quant_max={quant_max}, quant_min={quant_min}, qscheme={qscheme}, ada_sign={ada_sign}'
        return s.format(**self.__dict__)

    def forward(self, X):
        if self.training:
            if self.ada_sign and X.min() >= 0:
                self.quant_max = self.activation_post_process.quant_max = 2 ** self.bitwidth - 1
                self.quant_min = self.activation_post_process.quant_min = 0
                self.activation_post_process.adjust_sign = True

            self.activation_post_process(X.detach())

        _scale, _zero_point = self.activation_post_process.calculate_qparams()
        _zero_point.data.clamp_(self.quant_min, self.quant_max)

        if self.fake_quant_enabled[0] == 1:
            if self.pot_scale:
                _scale = pot_quantization(_scale)

            if is_per_channel(self.qscheme):
                X = torch.fake_quantize_per_channel_affine(
                    X, _scale, _zero_point.long(), self.ch_axis, self.quant_min, self.quant_max)
                # X = dsq_function_per_channel(
                #     X, _scale, _zero_point, self.quant_min, self.quant_max, self.ch_axis, self.alpha)
            else:
                X = torch.fake_quantize_per_tensor_affine(
                    X, float(_scale), int(_zero_point), self.quant_min, self.quant_max)
                # X = dsq_function_per_tensor(
                #     X, _scale, _zero_point,  self.quant_min, self.quant_max, self.alpha)

        return X
