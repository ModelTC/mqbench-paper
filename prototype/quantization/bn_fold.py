import torch
import torch.nn as nn
import torch.nn.functional as F
from prototype.utils.dist import allaverage_autograd


def freeze_bn_all(model):
    for mod in model.modules():
        if isinstance(mod, CustomSyncBN2d):
            mod.is_freezed = True


class CustomSyncBN2d(nn.BatchNorm2d):

    def __init__(self, bn):
        super().__init__(bn.num_features)
        self.weight.data = bn.weight.data.clone()
        self.bias.data = bn.bias.data.clone()
        self.running_var.data = bn.running_var.data.clone()
        self.running_mean.data = bn.running_mean.data.clone()
        self.eps = bn.eps
        self.momentum = bn.momentum
        self.freeze_step = 500001
        self.is_freezed = False

    def forward(self, x):
        x_mean, x_var = self.get_mean_var(x)
        x = (x - x_mean[None, :, None, None]) / (torch.sqrt(x_var[None, :, None, None] + self.eps))
        if self.affine:
            x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return x

    def get_mean_var(self, x):
        bn_training = self.training and not self.is_freezed
        if bn_training:
            x_data = x.transpose(0, 1).contiguous().view(self.num_features, -1)
            x_mean = x_data.mean(1)
            x_var = x_data.var(1, unbiased=False)

            # synchronize the batch statistics
            x_mean = allaverage_autograd(x_mean)
            x_var = allaverage_autograd(x_var)

            n = x_data.numel() / x_data.shape[1]
            self.running_mean = \
                (1. - self.momentum) * self.running_mean + \
                self.momentum * x_mean.data
            self.running_var = \
                (1. - self.momentum) * self.running_var + \
                self.momentum * x_var.data * n / (n - 1)
            self.num_batches_tracked = self.num_batches_tracked + 1
            if self.num_batches_tracked > self.freeze_step:
                self.is_freezed = True
        else:
            x_mean = self.running_mean
            x_var = self.running_var

        return x_mean, x_var

    def get_params(self):
        return self.running_mean.data, self.running_var.data, self.weight.data, self.bias.data


class ConvBNBase(nn.Module):
    """Quantized 2D convolution with batch norm folded.
    Baseline method introduced in integer-only-quantization
    """
    def __init__(self, conv_module: nn.Conv2d, bn_module: nn.BatchNorm2d, qconfig=None):
        super().__init__()
        self.qconfig = qconfig
        self.conv = conv_module
        self.bn = CustomSyncBN2d(bn_module)
        self.conv_params = {"stride": conv_module.stride,
                            "padding": conv_module.padding,
                            "dilation": conv_module.dilation,
                            "groups": conv_module.groups}
        self.weight_fake_quant = qconfig.weight()

    @classmethod
    def from_float(cls, mod):
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        qconfig = mod.qconfig
        conv, bn = mod[0], mod[1]
        qat_convbn = cls(conv, bn, qconfig)
        return qat_convbn

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBNReLUBase(ConvBNBase):

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ConvBNFreeze(ConvBNBase):

    def __init__(self, conv_module: nn.Conv2d, bn_module: nn.BatchNorm2d, qconfig=None):
        super().__init__(conv_module, bn_module, qconfig)
        self.bn.is_freezed = True

    def forward(self, input_tensor):
        y_mean, y_var = self.bn.running_mean, self.bn.running_var
        safe_std = torch.sqrt(y_var + self.bn.eps)
        weight, bias = self._fold_bn(self.conv.weight, y_mean, safe_std)
        weight = self.weight_fake_quant(weight)
        return F.conv2d(input_tensor, weight, bias, **self.conv_params)

    def _fold_bn(self, w, y_mean, safe_std):
        w_view = (self.bn.num_features, 1, 1, 1)
        weight = w * (self.bn.weight / safe_std).view(w_view)
        beta = self.bn.bias - self.bn.weight * y_mean / safe_std
        if self.conv.bias is not None:
            bias = self.bn.weight * self.conv.bias / safe_std + beta
        else:
            bias = beta

        return weight, bias


class ConvBNReLUFreeze(ConvBNFreeze):

    def forward(self, x):
        return F.relu(super().forward(x))


class ConvBNMerge(ConvBNFreeze):

    def __init__(self, conv_module: nn.Conv2d, bn_module: nn.BatchNorm2d, qconfig=None):
        super().__init__(conv_module, bn_module, qconfig)
        y_mean, y_var = self.bn.running_mean, self.bn.running_var
        safe_std = torch.sqrt(y_var + self.bn.eps)
        self.conv.weight.data, bias = self._fold_bn(self.conv.weight, y_mean, safe_std)
        self.conv.bias = nn.Parameter(bias)
        delattr(self, 'bn')

    def forward(self, input_tensor):
        weight, bias = self.conv.weight, self.conv.bias
        weight = self.weight_fake_quant(weight)
        return F.conv2d(input_tensor, weight, bias, **self.conv_params)


class ConvBNReLUMerge(ConvBNMerge):

    def forward(self, x):
        return F.relu(super().forward(x))


class ConvBNNaiveFold(ConvBNFreeze):

    def forward(self, input_tensor):
        out = F.conv2d(input_tensor, self.conv.weight, self.conv.bias, **self.conv_params)
        y_mean, y_var = self.bn.get_mean_var(out)
        safe_std = torch.sqrt(y_var + self.bn.eps)
        weight, bias = self._fold_bn(self.conv.weight, y_mean, safe_std)
        weight = self.weight_fake_quant(weight)
        return F.conv2d(input_tensor, weight, bias, **self.conv_params)


class ConvBNReLUNaiveFold(ConvBNNaiveFold):

    def forward(self, x):
        return F.relu(super().forward(x))


class ConvBNWPFold(ConvBNFreeze):

    def forward(self, input_tensor):
        bias_shape = [1] * len(self.conv.weight.shape)
        bias_shape[1] = -1

        bn_training = self.training and not self.bn.is_freezed
        out = F.conv2d(input_tensor, self.conv.weight, self.conv.bias, **self.conv_params)
        y_mean, y_var = self.bn.get_mean_var(out)

        safe_std = torch.sqrt(y_var + self.bn.eps)
        weight, bias = self._fold_bn(self.conv.weight, y_mean, safe_std)

        if bn_training:
            running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
            weight, _ = self._fold_bn(self.conv.weight, self.bn.running_mean, running_std)
            weight = self.weight_fake_quant(weight)
            # correct the output of the convolution
            corrected_weight = (running_std / safe_std).view(bias_shape)
            corrected_bias = bias.view(bias_shape)
            return F.conv2d(input_tensor, weight, None, **self.conv_params).mul_(corrected_weight).add_(corrected_bias)
        else:
            weight = self.weight_fake_quant(weight)
            return F.conv2d(input_tensor, weight, bias, **self.conv_params)


class ConvBNReLUWPFold(ConvBNWPFold):

    def forward(self, x):
        return F.relu(super().forward(x))


class ConvBNTorchFold(ConvBNBase):

    def forward(self, x):
        w_view = (self.bn.num_features, 1, 1, 1)
        bias_shape = [1] * len(self.conv.weight.shape)
        bias_shape[1] = -1
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        weight = self.conv.weight * scale_factor.view(w_view)
        # quantizing weight
        weight = self.weight_fake_quant(weight)
        out = F.conv2d(x, weight, None, **self.conv_params)
        out = out / scale_factor.reshape(bias_shape)
        return self.bn(out)


class ConvBNReLUTorchFold(ConvBNTorchFold):

    def forward(self, x):
        return F.relu(super().forward(x))


class ConvBNLookAheadFold(ConvBNBase):

    def __init__(self, conv_module: nn.Conv2d, bn_module: nn.BatchNorm2d, qconfig):
        super().__init__(conv_module, bn_module, qconfig)
        self.fold_cycle = 10
        self.is_freezed = False
        self.update_current_stats()

    def forward(self, input_tensor):
        w_view = (self.bn.num_features, 1, 1, 1)

        if self.is_freezed:
            weight = self.conv.weight * self.weight_mult.view(w_view)
            weight = self.weight_fake_quant(weight)
            return self.bn(F.conv2d(input_tensor, weight, self.beta, **self.conv_params))
        else:
            bn_training = self.training
            if bn_training:
                # check if we have to fold bn here
                if self.bn.num_batches_tracked > 0 and self.bn.num_batches_tracked % self.fold_cycle == 0:
                    self.update_current_stats()

                if self.bn.num_batches_tracked > self.bn.freeze_step:
                    self.update_current_stats()
                    self.is_freezed = True

            weight = self.conv.weight * self.weight_mult.view(w_view)
            weight = self.weight_fake_quant(weight)

            return self.bn(F.conv2d(input_tensor, weight, self.beta, **self.conv_params))

    def update_current_stats(self):
        mean, var, affine_weight, affine_bias = self.bn.get_params()
        safe_std = torch.sqrt(var + self.bn.eps)
        if not hasattr(self, 'weight_mult'):
            self.register_buffer('weight_mult', affine_weight / safe_std)
        else:
            self.weight_mult = self.weight_mult * (affine_weight / safe_std)
        if not hasattr(self, 'beta'):
            self.register_buffer('beta', affine_bias - affine_weight * mean / safe_std)
        else:
            self.beta = affine_bias + affine_weight * (self.beta - mean) / safe_std
        # set bn stats to affine
        self.bn.running_mean.data = self.bn.bias.data.clone()
        self.bn.running_var.data = self.bn.weight.data.clone() ** 2


class ConvBNReLULookAheadFold(ConvBNLookAheadFold):

    def forward(self, x):
        return F.relu(super().forward(x))


def search_fold_bn(model, strategy=4, **kwargs):

    def is_bn(m):
        return isinstance(m, nn.BatchNorm2d)

    def is_absorbing(m):
        return (isinstance(m, nn.Conv2d)) or isinstance(m, nn.Linear)

    model.eval()
    prev = None
    prev_name = 'conv'
    for n, m in model.named_children():
        if is_bn(m) and is_absorbing(prev):
            qconfig = prev.qconfig
            if strategy == -1:
                setattr(model, prev_name, ConvBNBase(prev, m, qconfig))
            elif strategy == 0:
                setattr(model, prev_name, ConvBNMerge(prev, m, qconfig))
            elif strategy == 1:
                setattr(model, prev_name, ConvBNFreeze(prev, m, qconfig))
            elif strategy == 2:
                setattr(model, prev_name, ConvBNNaiveFold(prev, m, qconfig))
            elif strategy == 3:
                setattr(model, prev_name, ConvBNWPFold(prev, m, qconfig))
            elif strategy == 4:
                setattr(model, prev_name, ConvBNTorchFold(prev, m, qconfig))
            setattr(model, n, nn.Identity())
        else:
            search_fold_bn(m, strategy, **kwargs)
        prev = m
        prev_name = n
