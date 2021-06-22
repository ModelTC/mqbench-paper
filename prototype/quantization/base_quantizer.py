# type: ignore
import torch
import math
from torch.nn.parameter import Parameter


class QuantizeBase(torch.quantization.FakeQuantizeBase):
    """
    Quantizer base class for each algorithm, need to specify args for observer, including
    :param quant_min: minimum quantization levels
    :param quant_max: maximum quantization levels
    :param ada_sign: using adaptive sign for activation quantization
    :param pot_scale: powers-of-2 scale
    :param dype: data type, default qint8
    :param qshcme: the quantizer scheme, sym or affine, per-channel or per-tensor
    :param ch_axis: channel axis
    """
    def __init__(self, observer, **observer_kwargs):
        super(QuantizeBase, self).__init__()
        self.activation_post_process = observer(**observer_kwargs)
        self.quant_min = self.activation_post_process.quant_min
        self.quant_max = self.activation_post_process.quant_max
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = self.activation_post_process.ch_axis \
            if hasattr(self.activation_post_process, 'ch_axis') else -1
        self.ada_sign = self.activation_post_process.ada_sign
        self.pot_scale = self.activation_post_process.pot_scale

        self.register_buffer('fake_quant_enabled', torch.tensor([1], dtype=torch.uint8))
        self.register_buffer('static_enabled', torch.tensor([1], dtype=torch.uint8))
        self.register_buffer('learning_enabled', torch.tensor([0], dtype=torch.uint8))

        bitrange = torch.tensor(self.quant_max - self.quant_min + 1).double()
        self.bitwidth = int(torch.log2(bitrange).item())
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))

    @torch.jit.export
    def enable_param_learning(self):
        r"""Enables learning of quantization parameters and
        disables static observer estimates. Forward path returns fake quantized X.
        """
        self.toggle_qparam_learning(enabled=True) \
            .toggle_fake_quant(enabled=True) \
            .toggle_observer_update(enabled=False)
        return self

    @torch.jit.export
    def enable_static_estimate(self):
        r"""Enables static observer estimates and disbales learning of
        quantization parameters. Forward path returns fake quantized X.
        """
        self.toggle_qparam_learning(enabled=False) \
            .toggle_fake_quant(enabled=True) \
            .toggle_observer_update(enabled=True)

    @torch.jit.export
    def enable_static_observation(self):
        r"""Enables static observer accumulating data from input but doesn't
        update the quantization parameters. Forward path returns the original X.
        """
        self.toggle_qparam_learning(enabled=False) \
            .toggle_fake_quant(enabled=False) \
            .toggle_observer_update(enabled=True)

    @torch.jit.export
    def toggle_observer_update(self, enabled=True):
        self.static_enabled[0] = int(enabled)
        return self

    @torch.jit.export
    def enable_observer(self, enabled=True):
        self.toggle_observer_update(enabled)

    @torch.jit.export
    def toggle_qparam_learning(self, enabled=True):
        self.learning_enabled[0] = int(enabled)
        return self

    @torch.jit.export
    def toggle_fake_quant(self, enabled=True):
        self.fake_quant_enabled[0] = int(enabled)
        return self

    @torch.jit.export
    def observe_quant_params(self):
        print('_LearnableFakeQuantize Scale: {}'.format(self.scale.detach()))
        print('_LearnableFakeQuantize Zero Point: {}'.format(self.zero_point.detach()))

    @torch.jit.export
    def calculate_qparams(self):
        self.scale.data.clamp_(min=self.eps.item())
        scale = self.scale.detach()
        zero_point = self.zero_point.detach().round().clamp(self.quant_min, self.quant_max).long()
        return scale, zero_point

    def extra_repr(self) -> str:
        s = 'bitwidth={bitwidth}, quant_max={quant_max}, quant_min={quant_min},' \
            'qscheme={qscheme}'
        return s.format(**self.__dict__)

# shutdown observation and fake quantize
def enable_param_learning(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase):
            module.enable_param_learning()

# start observation and fake quantize
def enable_static_estimate(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase):
            module.enable_static_estimate()

# start observation and original x
def enable_static_observation(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase):
            module.enable_static_observation()


def toggle_fake_quant(model, enabled=True):
    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase):
            module.toggle_fake_quant(enabled)


# a = torch.randn(1, 3, 224, 224)
# alpha = torch.tensor(1.0, requires_grad=True)
#
# import numpy as np
# a = np.array([72.4, 73.2, 0., 0., 72.2])
# print(a.mean(), a.std())
