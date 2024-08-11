from pytorch_quantization import nn as quant_nn
import os
import argparse
from mmengine import Config
import importlib
import sys
import torch

sys.path.append('/root/SparseOcc-TensorRT/')
from convert import pytorch2onnx

def atan2(g, self, other):
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

torch.onnx.register_custom_op_symbolic("atan2", atan2, 17)

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
    output_file = os.path.join(config.ONNX_PATH, output + ".onnx")

    pytorch2onnx(
        config,
        checkpoint=checkpoint_file,
        output_file=output_file,
        verbose=False,
        opset_version=args.opset_version,
        cuda=args.cuda,
    )
    print("ONNX Model has been saved in %s!"%(config.ONNX_PATH))

if __name__ == "__main__":
    main()
