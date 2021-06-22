# type: ignore
import torch
from .base_quantizer import QuantizeBase
from .utils import pot_quantization, grad_scale
from torch.nn.parameter import Parameter


class _AdditivePoTFakeQuantize(QuantizeBase):

    r""" This is an extension of the FakeQuantize module in fake_quantize.py, which
    supports more generalized lower-bit quantization and support learning of the scale
    and zero point parameters through backpropagation. For literature references,
    please see the class _LearnableFakeQuantizePerTensorOp.
    In addition to the attributes in the original FakeQuantize module, the _LearnableFakeQuantize
    module also includes the following attributes to support quantization parameter learning.
    """
    def __init__(self, observer, alpha=3.0, quantize_weight=True, **observer_kwargs):
        super(_AdditivePoTFakeQuantize, self).__init__(observer, **observer_kwargs)
        self.quantize_weight = quantize_weight
        if quantize_weight:
            self.alpha = Parameter(torch.tensor([alpha]))
        else:
            self.alpha = Parameter(torch.tensor([alpha * 2]))
        self.sign = True
        self.unsign_set = build_power_value(B=self.bitwidth)
        self.sign_set = build_power_value(B=self.bitwidth-1)

    @torch.jit.export
    def toggle_qparam_learning(self, enabled=True):
        self.learning_enabled[0] = int(enabled)
        self.alpha.requires_grad = enabled
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
            self.alpha.data.abs_()
            self.alpha.data.clamp_(min=1e-5)

            if self.bitwidth == 8:
                _scale, _zero_point = self.activation_post_process.calculate_qparams()
                X = torch.fake_quantize_per_tensor_affine(
                    X, float(_scale), int(_zero_point), self.quant_min, self.quant_max)
            else:
                if self.quantize_weight:
                    # weight normalization
                    X = (X - X.mean()) / (X.std() + 1e-5)
                X = apot_quantization(X, self.alpha, self.sign_set if self.sign else self.unsign_set, self.sign)

        return X


def build_power_value(B=4):
    base_a = [0.]
    base_b = [0.]
    base_c = [0.]
    if B == 2:
        for i in range(3):
            base_a.append(2 ** (-i - 1))
    elif B == 4:
        for i in range(3):
            base_a.append(2 ** (-2 * i - 1))
            base_b.append(2 ** (-2 * i - 2))
    elif B == 6:
        for i in range(3):
            base_a.append(2 ** (-3 * i - 1))
            base_b.append(2 ** (-3 * i - 2))
            base_c.append(2 ** (-3 * i - 3))
    elif B == 3:
        for i in range(3):
            if i < 2:
                base_a.append(2 ** (-i - 1))
            else:
                base_b.append(2 ** (-i - 1))
                base_a.append(2 ** (-i - 2))
    elif B == 5:
        for i in range(3):
            if i < 2:
                base_a.append(2 ** (-2 * i - 1))
                base_b.append(2 ** (-2 * i - 2))
            else:
                base_c.append(2 ** (-2 * i - 1))
                base_a.append(2 ** (-2 * i - 2))
                base_b.append(2 ** (-2 * i - 3))
    else:
        pass
    values = []
    for a in base_a:
        for b in base_b:
            for c in base_c:
                values.append((a + b + c))
    values = torch.Tensor(list(set(values)))
    values = values.mul(1.0 / torch.max(values))
    return values


def apot_quantization(tensor, alpha, proj_set, is_weight=True):
    def power_quant(x, value_s):
        if is_weight:
            shape = x.shape
            xhard = x.view(-1)
            sign = x.sign()
            value_s = value_s.type_as(x)
            xhard = xhard.abs()
            idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
            xhard = value_s[idxs].view(shape).mul(sign)
            xhard = xhard
        else:
            shape = x.shape
            xhard = x.view(-1)
            value_s = value_s.type_as(x)
            xhard = xhard
            idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
            xhard = value_s[idxs].view(shape)
            xhard = xhard
        xout = (xhard - x).detach() + x
        return xout

    data = tensor / alpha
    if is_weight:
        data = data.clamp(-1, 1)
        data_q = power_quant(data, proj_set)
        data_q = data_q * alpha
    else:
        data = data.clamp(0, 1)
        data_q = power_quant(data, proj_set)
        data_q = data_q * alpha
    return data_q