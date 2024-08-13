#!/bin/bash

gpu_id=0
while getopts "d:" opt
do
  case $opt in
    d)
      gpu_id=$OPTARG
      ;;
    ?)
      echo "There is unrecognized parameter."
      exit 1
      ;;
  esac
done

echo "Running on the GPU: $gpu_id"

CUDA_VISIBLE_DEVICES=$gpu_id python tools/onnx2trt.py \
configs/r50_nuimg_704x256_8f_trt.py \
checkpoints/onnx/sparseocc_r50_nuimg_704x256_8f_24e_v1_4.onnx \
--fp16
