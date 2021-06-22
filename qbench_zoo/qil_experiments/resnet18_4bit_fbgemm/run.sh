#!/usr/bin/env bash
PYTHONPATH=$PYTHONPATH:../../.. GLOG_vmodule=MemcachedClient=-1 \
spring.submit run --gpu --cpus-per-task 6 -n8 " python -u -m prototype.solver.cls_quant_solver --config config.yaml "
