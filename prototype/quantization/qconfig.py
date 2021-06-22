
import torch

from torch.quantization import MovingAverageMinMaxObserver, QConfig, FakeQuantize
from prototype.quantization.lsq import _LearnableFakeQuantize
from prototype.quantization.pact import _PACTFakeQuantize
from prototype.quantization.dorefa import _DoReFaFakeQuantize
from prototype.quantization.dsq import _DSQFakeQuantize
from prototype.quantization.apot import _AdditivePoTFakeQuantize
from prototype.quantization.qil import _QILQuantize
from prototype.quantization.ema import FixedFakeQuantize
from prototype.quantization.observer import BatchMeanMaxObserver, ClipStdObserver, QuantileMovingAverageObserver,\
    LSQObserver, MinMaxObserver, LSQPlusObserver, AverageLSQObserver, AverageMinMaxObserver, \
    KLDObserver, MSEObserver


def get_qconfig(w_method='lsq', a_method='lsq', bit=8, ada_sign=True, symmetry=True, per_channel=False, pot_scale=False,
                a_observer=None, w_observer=None):
    if per_channel:
        qscheme = torch.per_channel_symmetric if symmetry else torch.per_channel_affine
    else:
        qscheme = torch.per_tensor_symmetric if symmetry else torch.per_tensor_affine
    if symmetry is not True:
        # affine quantization does not need adaptive sign
        ada_sign = False

    fix_q_params = dict(
        quant_min=-2 ** (bit-1),
        quant_max=2 ** (bit-1) - 1,
        dtype=torch.qint8,
        pot_scale=pot_scale,
    )

    Observer_dict = {'MinMaxObserver': MinMaxObserver,
                     'ClipStdObserver': ClipStdObserver,
                     'AverageMinMaxObserver': AverageMinMaxObserver,
                     'LSQObserver': LSQObserver,
                     'LSQPlusObserver': LSQPlusObserver,
                     'AverageLSQObserver': AverageLSQObserver,
                     'QuantileMovingAverageObserver': QuantileMovingAverageObserver,
                     'MSEObserver': MSEObserver,
                     'KLDObserver': KLDObserver}

    if w_method == 'lsq':
        observer = Observer_dict[w_observer] if w_observer is not None else LSQObserver
        wq_config = _LearnableFakeQuantize.with_args(
            observer=observer,
            use_grad_scaling=True, ada_sign=False,
            qscheme=qscheme, ch_axis=0 if per_channel else -1,
            reduce_range=False, **fix_q_params)
    elif w_method == 'dorefa':
        observer = Observer_dict[w_observer] if w_observer is not None else MinMaxObserver
        wq_config = _DoReFaFakeQuantize.with_args(
            observer=observer,
            quantize_weight=True, ada_sign=False, ch_axis=0 if per_channel else -1,
            qscheme=qscheme, **fix_q_params)
    elif w_method == 'dsq':
        observer = Observer_dict[w_observer] if w_observer is not None else ClipStdObserver
        wq_config = _DSQFakeQuantize.with_args(
            observer=observer,
            ada_sign=False, ch_axis=0 if per_channel else -1,
            qscheme=qscheme, **fix_q_params)
    elif w_method == 'apot':
        observer = Observer_dict[w_observer] if w_observer is not None else MinMaxObserver
        wq_config = _AdditivePoTFakeQuantize.with_args(
            observer=observer,
            quantize_weight=True, ada_sign=False, ch_axis=0 if per_channel else -1,
            qscheme=qscheme, **fix_q_params)
    elif w_method == 'qil':
        observer = Observer_dict[w_observer] if w_observer is not None else MinMaxObserver
        wq_config = _QILQuantize.with_args(
            observer=observer,
            quantize_weight=True, ada_sign=False, ch_axis=0 if per_channel else -1,
            qscheme=qscheme, **fix_q_params)
    elif w_method == 'quantile':
        observer = Observer_dict[w_observer] if w_observer is not None else MovingAverageMinMaxObserver
        wq_config = FixedFakeQuantize.with_args(
            averaging_constant=1.0, quantile=0.999,
            observer=observer,
            use_grad_scaling=True, ada_sign=ada_sign,
            qscheme=torch.per_tensor_symmetric,
            reduce_range=False, **fix_q_params)
    else:
        raise NotImplementedError

    if a_method == 'lsq':
        observer = Observer_dict[a_observer] if a_observer is not None else LSQObserver
        aq_config = _LearnableFakeQuantize.with_args(
            observer=observer,
            use_grad_scaling=True, qscheme=torch.per_tensor_symmetric if symmetry else torch.per_tensor_affine,
            ada_sign=ada_sign, reduce_range=False, **fix_q_params)
    elif a_method == 'dorefa':
        observer = Observer_dict[a_observer] if a_observer is not None else MinMaxObserver
        aq_config = _DoReFaFakeQuantize.with_args(
            observer=observer,
            quantize_weight=False, ada_sign=ada_sign, ch_axis=-1,
            qscheme=torch.per_tensor_symmetric if symmetry else torch.per_tensor_affine, **fix_q_params)
    elif a_method == 'pact':
        observer = Observer_dict[a_observer] if a_observer is not None else MinMaxObserver
        aq_config = _PACTFakeQuantize.with_args(
            observer=observer,
            ada_sign=ada_sign, ch_axis=-1, alpha=6.0,
            qscheme=torch.per_tensor_symmetric if symmetry else torch.per_tensor_affine, **fix_q_params)
    elif a_method == 'dsq':
        observer = Observer_dict[a_observer] if a_observer is not None else BatchMeanMaxObserver
        aq_config = _DSQFakeQuantize.with_args(
            observer=observer,
            ada_sign=ada_sign, ch_axis=-1,
            qscheme=torch.per_tensor_symmetric if symmetry else torch.per_tensor_affine,
            **fix_q_params)
    elif a_method == 'apot':
        observer = Observer_dict[a_observer] if a_observer is not None else MinMaxObserver
        aq_config = _AdditivePoTFakeQuantize.with_args(
            observer=observer,
            quantize_weight=False, ada_sign=ada_sign, ch_axis=-1,
            qscheme=torch.per_tensor_symmetric if symmetry else torch.per_tensor_affine, **fix_q_params)
    elif a_method == 'qil':
        observer = Observer_dict[a_observer] if a_observer is not None else MinMaxObserver
        aq_config = _QILQuantize.with_args(
            observer=observer,
            quantize_weight=False, ada_sign=ada_sign, ch_axis=-1,
            qscheme=torch.per_tensor_symmetric if symmetry else torch.per_tensor_affine, **fix_q_params)
    elif a_method == 'ema':
        observer = Observer_dict[a_observer] if a_observer is not None else MovingAverageMinMaxObserver
        aq_config = FakeQuantize.with_args(
            observer=observer,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            quant_min=0,
            quant_max=2 ** bit - 1,
            averaging_constant=0.1,
        )
    elif a_method == 'quantile':
        observer = Observer_dict[a_observer] if a_observer is not None else QuantileMovingAverageObserver
        aq_config = FixedFakeQuantize.with_args(
            observer=observer,
            averaging_constant=0.1, quantile=0.999,
            use_grad_scaling=True, qscheme=torch.per_tensor_symmetric,
            ada_sign=ada_sign, reduce_range=False,
            **fix_q_params
        )
    else:
        raise NotImplementedError

    return QConfig(activation=aq_config, weight=wq_config)


def get_activation_fake_quantize(a_method):
    if a_method == 'lsq':
        aq_config = _LearnableFakeQuantize
    elif a_method == 'dorefa':
        aq_config = _DoReFaFakeQuantize
    elif a_method == 'apot':
        aq_config = _AdditivePoTFakeQuantize
    elif a_method == 'pact':
        aq_config = _PACTFakeQuantize
    elif a_method == 'dsq':
        aq_config = _DSQFakeQuantize
    elif a_method == 'ema':
        aq_config = FakeQuantize
    elif a_method == 'quantile':
        aq_config = FixedFakeQuantize
    else:
        raise NotImplementedError
    return aq_config
