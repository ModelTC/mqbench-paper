# MQBench: Towards Reproducible and Deployable Model Quantization Benchmark



We propose a benchmark to evaluate different quantization algorithms on various settings. MQBench is a first attempt to evaluate, analyze, and benchmark the reproducibility and deployability for model quantization algorithms. We choose multiple different platforms for real-world deployments, including CPU, GPU, ASIC, DSP, and evaluate extensive state-of-the-art quantization algorithms under a unified training pipeline. MQBench acts like a bridge to connect the algorithm and the hardware. We conduct a comprehensive analysis and find considerable intuitive or counter-intuitive insights.



## Table of Contents

- [Table of Contents](https://github.com/TheGreatCold/mqbench#table-of-contents)
- [Highlighted Features](https://github.com/TheGreatCold/mqbench#highlighted-features)
- [Installation](https://github.com/TheGreatCold/mqbench#installation)
- [How to self-implement a quantization algorithm](https://github.com/TheGreatCold/mqbench#how-to-self-implement-a-quantization-algorithm)
- [How to self-implement a hardware configuration](https://github.com/TheGreatCold/mqbench#how-to-self-implement-a-hardware-configuration)
- [Submitting Your Results to MQBench](https://github.com/TheGreatCold/mqbench#submitting-your-results-to-mqbench)
- [Lisence](https://github.com/TheGreatCold/mqbench#license)





## Highlighted Features

+ Integrate with the latest tracing techniques in [Pytorch](https://pytorch.org/) 1.8.

+ Quantization Algorithms

  + Learned Step Size Quantization: https://arxiv.org/abs/1902.08153
  + Quantization Interval Learning: https://arxiv.org/abs/1808.05779
  + Differentiable Soft Quantization: https://arxiv.org/abs/1908.05033
  + Parameterized Clipping AcTivation: https://arxiv.org/abs/1805.06085
  + Additive Powers-of-Two Quantization: https://arxiv.org/abs/1909.13144
  + DoReFa-Net: https://arxiv.org/abs/1606.06160

+ Network Architectures:

  + ResNet-18, ResNet-50: https://arxiv.org/abs/1512.03385
  + MobileNetV2: https://arxiv.org/abs/1801.04381
  + EfficienteNet-Lite-B0: https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html
  + RegNetX-600GF: https://arxiv.org/abs/2003.13678

+ Hardware Platform:

  | Library  | Haware Type | s Form | Granularity | Symmetry   | Fold  BN |
  | -------- | ----------- | ------ | ----------- | ---------- | -------- |
  | Academic | None        | FP32   | Per-tensor  | Symmetric  | No       |
  | TensorRT | GPU         | FP32   | Per-channel | Symmetric  | Yes      |
  | ACL      | ASIC        | FP32   | Per-channel | Asymmetric | Yes      |
  | TVM      | ARM CPU     | POT    | Per-tensor  | Symmetric  | Yes      |
  | SNPE     | DSP         | FP32   | Per-tensor  | Asymmetric | Yes      |
  | FBGEMM   | X86 CPU     | FP32   | Per-channel | Asymmetric | Yes      |



## Installation

These instructions will help get MQBench up.

1. Clone MQBench.

2. (Optionally) Create a Python virtual environment.

3. Install the MQBench-required packages

   `$ pip install -r requirements.txt`

   Notes: MQBench uses Pytorch-1.8, our quantized model is based on the new `torch.fx` tracing techniques.

4. MQBench use the Pytorch distributed data-parallel training with `nccl` backend (see details [here](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)), please make sure your machine can initailize that distributed learning environment. 

## How to Reproduce MQBench

We provide the running scripts `run.sh` and configuration file `config.yaml` of all experiments in MQBench. 

To reproduce LSQ on ResNet-18, 

1. enter the directory 

   ```
   $ cd PATH-TO-PROJECT/qbench_zoo
   $ cd lsq_experiments/resnet18_4bit_academic
   ```

2. run script

   ```
   $ sh run.sh
   ```

   Note that `run.sh` contain some commands that may not be found, the core running command is

   ```bash
   PYTHONPATH=$PYTHONPATH:../../..
   python -u -m prototype.solver.cls_quant_solver --config config.yaml
   ```



## How to self-implement a quantization algorithm

All our quantization algorithms are implemented in `prototype/quantization/`

To implementa a new algorithm, you need to add you quantizer into this directory. 

All quantizer are inheritant from  `QuantizeBase` class. Each `QuantizedBase` will have an observer class which is used to estimate/update the quantization range. The observer design is inspired from the Pytorch-1.8 [repo](https://github.com/pytorch/pytorch/blob/master/torch/quantization/observer.py). Intializing a `QuantizeBase` class will also initialize a `Observer` class. 

The parameters contained for  `QuantizeBase` and `Observer` include：

1. `quant_min, quant_max`, which specify the $N_{min}, N_{max}$ for rounding boundaries. 
2. `qshcme`, which can be `torch.per_tensor_symmetric`, `torch.per_channel_symmetric`, `torch.per_tensor_affine`, and `torch.per_channel_affine`. This is often determined by the hardware setup. 
3. `ch_axis`, which is the dimension of channel-wise quantization. -1 is for per-tensor quantization. Typically for `nn.Conv2d` and `nn.Linear` module, the `ch_axis` should be 0.
4. `ada_sign`, which can adaptively choose the signness. `ada_sign` should be enabled for academic setting only.
5. `pot_scale`, which is used to determine the powers-of-two scale parameters.  

Note: each specified quantizer may have its own unique parameters, see example of LSQ below.

**Example Implementation of LSQ:**

1. For initialization, we add new parameters for storing the scale, zero_point:

   ```python
   self.use_grad_scaling = use_grad_scaling
   self.scale = Parameter(torch.tensor([scale]))
   self.zero_point = Parameter(torch.tensor([zero_point]))
   ```

2. The major implementation is the `forward` function, which should contain several cases:

   1. In case of `ada_sign=True`, the quantization range should be adjusted. 

      ```python
      if self.ada_sign and X.min() >= 0:
        	self.quant_max = self.activation_post_process.quant_max = 2 ** self.bitwidth - 1
        	self.quant_min = self.activation_post_process.quant_min = 0
        	self.activation_post_process.adjust_sign = True
      ```

   2. In case of symmetric quantization, the zero point should set to 0.

      ```python
      self.zero_point.data.zero_()
      ```

   3. In case of powers-of-two scale, the scale should be quantized by:

      ```python
      def pot_quantization(tensor: torch.Tensor):
          log2t = torch.log2(tensor)
          log2t = (torch.round(log2t)-log2t).detach() + log2t
          return 2 ** log2t
          
      scale = pot_quantization(self.scale)
      ```

   4. Implement both per-channel and per-tensor quantization.

**After adding you quantizer...**

The next step is to register the quantizer in `prototype/quantization/qconfig.py` 

Import your quantizer and then add it to `get_qconfig` function, and parse necessary arguments. 

The final step is to override a `config.yaml` file:

```yaml
qparams:
    w_method: lsq
    a_method: lsq
    bit: 4

backend: academic
bnfold: 4
```

By replacing the `w_method, a_method`, you can run your implementation. 

Note: the rest of the config file should not be modified in order to keep a unified training setting. 

How to self-implement a hardware configuration
-----------------------------------------------

Adding a new setting in hardware is much simpler that algorithms. To do this, we can add another condition in the ``if-else`` selection. For example, adding a new hardware TFLite Micro:


```python
        elif backend == "tflitemicro":
            backend_params = dict(ada_sign=False, symmetry=True, per_channel=False, pot_scale=True)
        ...

    model_qconfig = get_qconfig(**self.qparams, **backend_params)
    model = quantize_fx.prepare_qat_fx(model, {"": model_qconfig}, foldbn_config)
```

## Submitting Your Results to MQBench

You can submit your implementation to MQBench by submmitting a merge request to this repo. The implementation of new algorithms and the running scripts, log file are needed for evalutation. 



## License

This project is under Apache 2.0 License. 

