Quantization Hardware
========================================

.. _TensorRT: https://github.com/NVIDIA/TensorRT
.. _TVM: https://www.usenix.org/conference/osdi18/presentation/chen
.. _SNPE: https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk
.. _ACL: https://support.huaweicloud.com/intl/en-us/ti-mc-A200_3000/altasmodelling_16_043.html
.. _FBGEMM: https://github.com/pytorch/FBGEMM


Overview
--------

=================  ========  =======  ===========  ==============  ===========  ========  =====  ===
Inference Library  Provider  HW Type  Hardware     :math:`s` Form  Granularity  Symmetry  Graph  FBN
=================  ========  =======  ===========  ==============  ===========  ========  =====  ===
Academic           None      None     None         FP32            Per-tensor   Sym.      1      No
`TensorRT`_        NVIDIA    GPU      Tesla T4/P4  FP32            Per-channel  Sym.      2      Yes
`ACL`_             HUAWEI    ASIC     Ascend310    FP32            Per-channel  Asym.     1      Yes
`TVM`_             OctoML    CPU      ARM          POT             Per-tensor   Sym.      3      Yes
`SNPE`_            Qualcomm  DSP      Snapdragon   FP32            Per-tensor   Asym.     3      Yes
`FBGEMM`_          Facebook  CPU      X86          FP32            Per-channel  Asym.     3      Yes
=================  ========  =======  ===========  ==============  ===========  ========  =====  ===


Academic Setup
--------------


In academical research, most existing work chooses the per-tensor, symmetric quantization. This quantizer design could be challenging. However, academical setting only quantizes the input and the weight of a convolutional or linear layer. Thus the computational graph is not aligned to any hardware implementations. Note that people tend to use *unsigned* quantization to quantize input activation and *signed* quantization to quantize weights. For unsigned quantization, the target integer range is :math:`[N_{min}, N_{max}]=[0, 2^t-1]`, while for signed quantization, the range becomes :math:`[N_{min}, N_{max}]=[-2^{t-1}, 2^{t-1}-1]`. 
The intuition for adopting unsigned quantization is ReLU activation are non-negative, and symmetric signed quantization will waste one bit for negative parts. 
In our implementation, we add a switch variable called *adaptive signness*, which can turn the integer range to :math:`[-2^{t-1}, 2^{t-1}-1]` based on data statistics. It should be noted that *Adaptive signness* is only designed for academic setting, while symmetric quantization must waste one bit for non-negative activation in real-world hardware.


TensorRT Setup
--------------


`TensorRT`_ is a high-performance inference library developed by NVIDIA. The quantization scheme in TensorRT is symmetric per-channel for weights, and symmetric per-tensor for activations. The integer range is :math:`[-128, 127]`. TensorRT will quantize every layers in the network including the elemental-wise Add and Pooling besides those layers which have weights. However, per-channel quantization scheme can reduce the error for weight quantization to a certain extent. TensorRT model will be deployed on GPUs, and Int8 computation will be achieved by Tensor Cores or DP4A instructions, which are highly efficient. Typically, one GTX1080TI GPU have 45.2 peak Tops in INT8 mode.

SNPE Setup
----------

`SNPE`_ is a neural processing engine SDK developed by Qualcomm Snapdragon for model inference on Snapdragon CPU, Adreno GPU and the Hexagon DSP. SNPE supports 8bit fixed-point quantization for Hexagon DSP. Hexagon DSP is an advanced, variable instruction lenghth, Very Long Instruction Word(VLIW) processor architecture with hardware multi-threading. The quantization scheme of SPNE is asymmetric and per-tensor for weights and activations.


TVM Setup
---------

`TVM`_ is a deep learning compiler and can compile neural networks to a 8-bit implementation. For now, TVM supports running the whole network in symmetric per-tensor quantization scheme. One different point is, in order to accelerate the quantization affine operation, they represent the scale as power-of-two and thus can utilize the efficient shift operation to enjoy further speed up. The quantized INT8 model compiled by TVM can be deployed on GPUs, CPUs or DSPs.

ACL Setup
----------

`ACL`_ is a neural network inference software developed by HUAWEI for the hardware named Atlas. Atlas supports INT8 convolution and linear kernel so ACL quantizes the layer which has weights, such as convolution, fully-connected layer to int8 fixed point, but remains the rest part of network in FP32. The quantization scheme is symmetric per-channel for weight, and asymmetric per-tensor for activation to avoid the waste of one bit. Typically an Atlas 300 inference card have 88 Tops in INT8 mode.

FBGEMM Setup
------------

`FBGEMM`_ is a inference library developed by Facebook and can deploy torch model easily. The quantization scheme is asymmetric per-channel and we quantize the whole network into int8 fixed point.
