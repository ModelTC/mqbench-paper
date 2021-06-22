#!/usr/bin/env bash
export PATH=~/.local/bin/:$PATH
export LD_LIBRARY_PATH=/mnt/lustre/share/cuda-10.2/lib64:$LD_LIBRARY_PATH
PYTHONPATH=$PYTHONPATH:../../../.. GLOG_vmodule=MemcachedClient=-1 \
spring.submit run --gpu --cpus-per-task 4 -n8 \
" python -u -m prototype.solver.cls_quant_ptq_solver --config config.yaml --a_observer $1 --w_observer $2"
