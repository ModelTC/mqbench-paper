.. MQBench documentation master file, created by
   sensetime-research on Thu Jul 8 17:28:35 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MQBench's documentation!
========================================


What is MQBench?
========================================

MQBench is short for Model Quantization Benchmark, an open-source Python package for evalutaing quantization algorithoms.

MQBench tries to solve two long-neglected problems in the quantization community, namely reproducibility and deployability. 
For reproducibility, we setup a unified set of training hyper-parameters and run all algorithms under this setting. For deployability, we ensure every algorithms can be deployed, this includes the folding of Batch Normalization layers and corresponding hardware quantizer design. 

Key Features in MQBench:

+ Implementation with the lastest Pytorch-1.8 tracing techniques, easy to extend.
+ Support for 6 quantization-aware training algorithms and is easily extendable.
+ Support 5 hardware platforms.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   algorithm/index
   hardware/index
   benchmark/index


