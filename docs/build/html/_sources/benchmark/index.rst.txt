Benchmark
========================================

.. _here: https://pytorch.org/docs/stable/fx.html
.. _repo: https://github.com/pytorch/pytorch/blob/master/torch/quantization/observer.py


Running experiments in MQBench
------------------------------


MQBench integrate the lastest quantization features in Pytorch. With the help of ``torch.fx``, we can automated trace a model and get its computation graph.

	FX is a toolkit for developers to use to transform nn.Module instances. FX consists of three main components: a symbolic tracer, an intermediate representation, and Python code generation.

	The symbolic tracer performs “symbolic execution” of the Python code. It feeds fake values, called Proxies, through the code. Operations on theses Proxies are recorded. More information about symbolic tracing can be found in the symbolic_trace() and Tracer documentation.

	The intermediate representation is the container for the operations that were recorded during symbolic tracing. It consists of a list of Nodes that represent function inputs, callsites (to functions, methods, or torch.nn.Module instances), and return values. More information about the IR can be found in the documentation for Graph. The IR is the format on which transformations are applied.

	Python code generation is what makes FX a Python-to-Python (or Module-to-Module) transformation toolkit. For each Graph IR, we can create valid Python code matching the Graph’s semantics. This functionality is wrapped up in GraphModule, which is a torch.nn.Module instance that holds a Graph as well as a forward method generated from the Graph.

	Taken together, this pipeline of components (symbolic tracing -> intermediate representation -> transforms -> Python code generation) constitutes the Python-to-Python transformation pipeline of FX. In addition, these components can be used separately. For example, symbolic tracing can be used in isolation to capture a form of the code for analysis (and not transformation) purposes. Code generation can be used for programmatically generating models, for example from a config file. There are many uses for FX!

See details of ``torch.fx`` `here`_. 


Our framework adopts the official API to convert a quantized model. 
First, the backend parameters need to be determined. This includes the symmetry, per-channel or per-tensor quantization and POT or FP32 scale. Take SNPE as an example: the backend parameters are defined as follows:

.. code-block:: python

	backend_params = dict(ada_sign=False, symmetry=False, per_channel=False, pot_scale=False)

Then, we can convert the model with the official API:

.. code-block:: python
	
	import torch.quantization.quantize_fx as quantize_fx

	model = model_entry(self.config.model, pretrained=True)
	model_qconfig = get_qconfig(**self.qparams, **backend_params)
	foldbn_config = get_foldbn_config(foldbn_strategy)
	model = quantize_fx.prepare_qat_fx(model, {"": model_qconfig}, foldbn_config)

Here ``model_qconfig`` and ``foldbn_config`` is determined by the quantization algorithms. Users can freely choose the strategy for BN folding and the algorithms for QAT. After the ``prepare_qat_fx``, the model can be trained as a normal Pytorch ``nn.Module``. 


We provide the running scripts `run.sh` and configuration file `config.yaml` of all experiments in MQBench. 

To reproduce LSQ on ResNet-18, 

1. enter the directory 

.. code-block:: bash

   cd PATH-TO-PROJECT/qbench_zoo
   cd lsq_experiments/resnet18_4bit_academic

2. run script

.. code-block:: bash

   sh run.sh


Note that ``run.sh`` contain some commands that may not be found, the core running command is

.. code-block:: bash

   PYTHONPATH=$PYTHONPATH:../../..
   python -u -m prototype.solver.cls_quant_solver --config config.yaml


How to self-implement a quantization algorithm
----------------------------------------------

All our quantization algorithms are implemented in `prototype/quantization/`

To implementa a new algorithm, you need to add you quantizer into this directory. 

All quantizer are inheritant from ``QuantizeBase`` class. Each ``QuantizedBase`` will have an observer class which is used to estimate/update the quantization range. The observer design is inspired from the Pytorch-1.8 `repo`_. By intializing a ``QuantizeBase`` class, a ``Observer`` class will also be initialized to *observe* the activation range for this quantization node. 

The parameters contained for  ``QuantizeBase`` and ``Observer`` include：

1. ``quant_min, quant_max``, which specify the :math:`N_{min}, N_{max}` for rounding boundaries. 
2. ``qshcme``, which can be ``torch.per_tensor_symmetric``, ``torch.per_channel_symmetric``, ``torch.per_tensor_affine``, and ``torch.per_channel_affine``. This is often determined by the hardware setup. 
3. ``ch_axis``, which is the dimension of channel-wise quantization. -1 is for per-tensor quantization. Typically for ``nn.Conv2d`` and ``nn.Linear`` module, the ``ch_axis`` should be 0.
4. ``ada_sign``, which can adaptively choose the signness. ``ada_sign`` should be enabled for academic setting only.
5. ``pot_scale``, which is used to determine the powers-of-two scale parameters.  

Note: each specified quantizer may have its own unique parameters, see example of LSQ below.

**Example Implementation of LSQ:**

1. For initialization, we add new parameters for storing the scale, zero_point:

.. code-block:: python

   self.use_grad_scaling = use_grad_scaling
   self.scale = Parameter(torch.tensor([scale]))
   self.zero_point = Parameter(torch.tensor([zero_point]))

2. The major implementation is the `forward` function, which should contain several cases:

   1. In case of `ada_sign=True`, the quantization range should be adjusted. 

.. code-block:: python

	      if self.ada_sign and X.min() >= 0:
	        	self.quant_max = self.activation_post_process.quant_max = 2 ** self.bitwidth - 1
	        	self.quant_min = self.activation_post_process.quant_min = 0
	        	self.activation_post_process.adjust_sign = True
      

   2. In case of symmetric quantization, the zero point should set to 0.

.. code-block:: python

      self.zero_point.data.zero_()

   3. In case of powers-of-two scale, the scale should be quantized by:

.. code-block:: python

      def pot_quantization(tensor: torch.Tensor):
          log2t = torch.log2(tensor)
          log2t = (torch.round(log2t)-log2t).detach() + log2t
          return 2 ** log2t
          
      scale = pot_quantization(self.scale)

   4. Implement both per-channel and per-tensor quantization.

**After adding you quantizer...**

The next step is to register the quantizer in `prototype/quantization/qconfig.py` 

Import your quantizer and then add it to `get_qconfig` function, and parse necessary arguments. 

The final step is to override a `config.yaml` file:

.. code-block:: yaml

	qparams:
	    w_method: lsq
	    a_method: lsq
	    bit: 4

	backend: academic
	bnfold: 4

By replacing the ``w_method, a_method``, you can run your implementation. 

Note: the rest of the config file should not be modified in order to keep a unified training setting. 


How to self-implement a hardware configuration
-----------------------------------------------

Adding a new setting in hardware is much simpler that algorithms. To do this, we can add another condition in the ``if-else`` selection. For example, adding a new hardware TFLite Micro:


.. code-block:: python


            elif backend == "tflitemicro":
                backend_params = dict(ada_sign=False, symmetry=True, per_channel=False, pot_scale=True)
            ...

        model_qconfig = get_qconfig(**self.qparams, **backend_params)
        model = quantize_fx.prepare_qat_fx(model, {"": model_qconfig}, foldbn_config)

