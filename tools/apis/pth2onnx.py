from pytorch_quantization import nn as quant_nn
import os
import argparse
from mmcv import Config

import sys

sys.path.append(".")
from ...sparseocc_onnx_trt.convert import pytorch2onnx


def parse_args():
    parser = argparse.ArgumentParser(description="Convert PyTorch to ONNX")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--opset_version", type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--flag", default="", type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config_file = args.config
    checkpoint_file = args.checkpoint

    config = Config.fromfile(config_file)
    if hasattr(config, "plugin"):
        import importlib
    
    # register custom module
    importlib.import_module('...third_party')
    importlib.import_module('...loaders')


    output = os.path.split(args.checkpoint)[1].split(".")[0]

    if args.int8:
        quant_nn.TensorQuantizer.use_fb_fake_quant = True
    if args.flag:
        output += f"_{args.flag}"
    output_file = os.path.join(config.ONNX_PATH, output + ".onnx")

    pytorch2onnx(
        config,
        checkpoint=checkpoint_file,
        output_file=output_file,
        verbose=False,
        opset_version=args.opset_version,
        cuda=args.cuda,
    )


if __name__ == "__main__":
    main()
