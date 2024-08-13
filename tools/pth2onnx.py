from pytorch_quantization import nn as quant_nn
import os
import argparse
from mmengine import Config
import importlib
import sys
import torch
import functools
sys.path.append('/root/SparseOcc-TensorRT/')
from convert import pytorch2onnx
from torch.onnx import _type_utils, symbolic_helper
from torch.onnx._internal import _beartype, jit_utils, registration
from torch import _C
from torch.nn.functional import (
    GRID_SAMPLE_INTERPOLATION_MODES,
    GRID_SAMPLE_PADDING_MODES,
)
import math
import torch.nn.functional as F
from typing import Sequence

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=17)

@_onnx_symbolic("aten::atan2")
def atan2(g: jit_utils, self, other):
    # self is y, and other is x on coordinate
    slope = g.op("Div", self, other)
    atan = g.op("Atan", slope)
    const_zero = g.op("Constant", value_t=torch.tensor(0))
    const_pi = g.op("Constant", value_t=torch.tensor(math.pi))

    condition_second_or_third_quadrant = g.op("Greater", self, const_zero)
    second_third_quadrant = g.op(
        "Where",
        condition_second_or_third_quadrant,
        g.op("Add", atan, const_pi),
        g.op("Sub", atan, const_pi),
    )

    condition_14_or_23_quadrant = g.op("Less", other, const_zero)
    result = g.op("Where", condition_14_or_23_quadrant, second_third_quadrant, atan)

    return result

def convert_grid_sample_mode(mode_s):
    return (
        "linear" if mode_s == "bilinear" else "cubic" if mode_s == "bicubic" else mode_s
    )
    
@_onnx_symbolic("aten::grid_sampler")
@symbolic_helper.parse_args("v", "v", "i", "i", "b")
def grid_sampler(
    g: jit_utils.GraphContext,
    input: _C.Value,
    grid: _C.Value,
    mode_enum: int,
    padding_mode_enum: int,
    align_corners: bool,
):
    
    mode_s = {v: k for k, v in F.GRID_SAMPLE_INTERPOLATION_MODES.items()}[mode_enum]  # type: ignore[call-arg, index]
    # mode string changes at https://onnx.ai/onnx/operators/text_diff_GridSample_16_20.html
    mode_s = convert_grid_sample_mode(mode_s)
    padding_mode_s = {v: k for k, v in F.GRID_SAMPLE_PADDING_MODES.items()}[padding_mode_enum]  # type: ignore[call-arg, index]
    return g.op(
        "GridSample",
        input,
        grid,
        align_corners_i=int(align_corners),
        mode_s=mode_s,
        padding_mode_s=padding_mode_s,
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Convert PyTorch to ONNX")
    parser.add_argument("--config", help="test config file path")
    parser.add_argument("--checkpoint", help="checkpoint file")
    parser.add_argument("--int8", default=False, type=bool)
    parser.add_argument("--opset_version", default=17, type=int)
    parser.add_argument("--cuda", default=True, type=bool)
    parser.add_argument("--flag", default="", type=str)
    args = parser.parse_args()
    return args

# @_onnx_symbolic("aten::layer_norm")
# @symbolic_helper.parse_args("v", "is", "v", "v", "f", "none")
# def layer_norm(
#     g: jit_utils.GraphContext,
#     input: _C.Value,
#     normalized_shape: Sequence[int],
#     weight: _C.Value,
#     bias: _C.Value,
#     eps: float,
#     cudnn_enable: bool,
# ):
#     # normalized_shape: input shape from an expected input of size
#     # axis: The first normalization dimension.
#     # layer_norm normalizes on the last D dimensions,
#     # where D is the size of normalized_shape
#     axis = -len(normalized_shape)
#     return g.op(
#         "LayerNorm",
#         input,
#         weight,
#         bias,
#         epsilon_f=eps,
#         axis_i=axis,
#     )
    
def main():
    args = parse_args()

    config_file = args.config
    checkpoint_file = args.checkpoint

    config = Config.fromfile(config_file)
    
    # register custom module
    importlib.import_module('third_party')
    importlib.import_module('loaders')
    importlib.import_module('sparseocc_onnx_trt')

    output = os.path.split(args.checkpoint)[1].split(".")[0]

    if args.int8:
        quant_nn.TensorQuantizer.use_fb_fake_quant = True
    if args.flag:
        output += f"_{args.flag}"
    output_file = os.path.join(config.ONNX_PATH, output + "_4.onnx")

    pytorch2onnx(
        config,
        checkpoint=checkpoint_file,
        output_file=output_file,
        verbose=True,
        opset_version=args.opset_version,
        cuda=args.cuda,
    )
    print("ONNX Model has been saved in %s!"%(config.ONNX_PATH))

if __name__ == "__main__":
    main()
