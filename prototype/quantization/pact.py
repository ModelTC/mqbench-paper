# type: ignore
import torch
import spring.linklink as link
from torch.nn.parameter import Parameter
from .base_quantizer import QuantizeBase
from .utils import pot_quantization, is_symmetric


class _PACTFakeQuantize(QuantizeBase):
    def __init__(self, observer, alpha=6.0, **observer_kwargs):
        super(_PACTFakeQuantize, self).__init__(observer, **observer_kwargs)
        self.alpha = Parameter(torch.tensor([alpha]))
        if not is_symmetric(self.qscheme):
            self.n_alpha = Parameter(torch.tensor([-alpha]))

    def extra_repr(self) -> str:
        s = 'bitwidth={bitwidth}, quant_max={quant_max}, quant_min={quant_min}, qscheme={qscheme}, ada_sign={ada_sign}'
        return s.format(**self.__dict__)

    def forward(self, X):
        if 1:
            if self.ada_sign and X.min() >= 0:
                self.quant_max = self.activation_post_process.quant_max = 2 ** self.bitwidth - 1
                self.quant_min = self.activation_post_process.quant_min = 0
                self.activation_post_process.adjust_sign = True

            self.activation_post_process(X.detach())
            if self.bitwidth != 8:
                self.alpha.data = self.alpha.data.abs()
                X = torch.where(X > self.alpha, self.alpha, X)
                self.activation_post_process.max_val.data.fill_(self.alpha.data[0])
                if X.min() < 0:
                    if is_symmetric(self.qscheme):
                        X = torch.where(X < -self.alpha, -self.alpha, X)
                        self.activation_post_process.min_val.data.fill_(-self.alpha[0].data)
                    else:
                        X = torch.where(X < self.n_alpha, self.n_alpha, X)
                        self.activation_post_process.min_val.data.fill_(self.n_alpha[0].data)
                else:
                    self.activation_post_process.min_val.data.fill_(0.)

            _scale, _zero_point = self.activation_post_process.calculate_qparams()
            # self.scale = _scale
            # self.zero_point = _zero_point

        if self.fake_quant_enabled[0] == 1:
            if self.pot_scale:
                _scale = pot_quantization(_scale)
            # if is_symmetric(self.qscheme):
            #     _zero_point = 0
            X = torch.fake_quantize_per_tensor_affine(
                X, float(_scale), int(_zero_point), self.quant_min, self.quant_max)

        return X

