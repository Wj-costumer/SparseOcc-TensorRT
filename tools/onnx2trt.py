# import tensorrt as trt
# import ctypes
# import numpy as np
# from mmdeploy.backend.tensorrt import load_tensorrt_plugin

# def make_network_and_engine(logger: trt.Logger, 
#                             plugin: trt.IPluginV2,
#                             input_shape: tuple, 
#                             grid_shape: tuple,
#                             precision = "float32"):
    
#     builder = trt.Builder(logger)
#     explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
#     network = builder.create_network(explicit_batch)
#     # network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
#     config  = builder.create_builder_config()
#     runtime = trt.Runtime(logger)

#     if precision == "float32":
#         input_layer = network.add_input(name="input", dtype=trt.float32, shape=input_shape)
#         grid_layer = network.add_input(name="grid", dtype=trt.float32, shape=grid_shape)
#     elif precision == "float16":
#         input_layer = network.add_input(name="input", dtype=trt.float16, shape=input_shape)
#         grid_layer = network.add_input(name="grid", dtype=trt.float16, shape=grid_shape)
#         config.set_flag(trt.BuilderFlag.FP16)
#     else:
#         raise Exception("Unsupported: {}".format(precision))

#     grid_sample_layer = network.add_plugin_v2(inputs=[input_layer, grid_layer], plugin=plugin)
#     print(type(grid_sample_layer))
#     print(type(grid_sample_layer.get_output(0)))
#     network.mark_output(grid_sample_layer.get_output(0))

#     engine_string = builder.build_serialized_network(network, config)
#     engine = runtime.deserialize_cuda_engine(engine_string)

#     return engine

# # load_tensorrt_plugin()

# onnx_path = "./sparseocc_gs.onnx"
# engine_path = "./sparseocc_trt.trt"

# logger  = trt.Logger(trt.Logger.VERBOSE)
# handle=ctypes.CDLL("third_party/plugin/grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so", mode = ctypes.RTLD_GLOBAL)

# trt.init_libnvinfer_plugins(logger, "")
# # trt.get_plugin_registry().load_library('third_party/plugin/grid-sample3d-trt-plugin')
# if not handle:
#     print("load grid_sample_3d plugin error")

# builder = trt.Builder(logger)
# explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
# network = builder.create_network(explicit_batch)
# # network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

# parser  = trt.OnnxParser(network, logger)
# config  = builder.create_builder_config()


# # registry = trt.get_plugin_registry()
# # plugin_creator = registry.get_plugin_creator("GridSample3D", "1", "")
# # breakpoint()
# # pf_interpolation_mode = trt.PluginField("interpolation_mode", np.array([0], np.int32), trt.PluginFieldType.INT32)
# # pf_padding_mode = trt.PluginField("padding_mode", np.array([0], np.int32), trt.PluginFieldType.INT32)
# # pf_align_corners = trt.PluginField("align_corners", np.array([0], np.int32), trt.PluginFieldType.INT32)
# # pfc = trt.PluginFieldCollection([pf_interpolation_mode, pf_padding_mode, pf_align_corners])
# # plugin = plugin_creator.create_plugin("grid_sample_3d", pfc)


# with open(onnx_path, "rb") as f:
#     if not parser.parse(f.read()):
#         print("ERROR: Failed to parse the ONNX file.")
#         for error in range(parser.num_errors):
#             print(parser.get_error(error))
#     parser.parse(f.read())
# # last_layer = network.get_layer(network.num_layers - 1)
# # network.mark_output(last_layer.get_output(0))
# breakpoint()
# engineString = builder.build_serialized_network(network, config)
# # input_shape = (1, 32, 16, 64, 64)
# # grid_shape = (1, 16, 64, 64, 3)
# # engine = make_network_and_engine(logger, plugin, input_shape, grid_shape, "float16"
    
# runtime = trt.Runtime(logger) 
# engine = runtime.deserialize_cuda_engine(engineString)
# with open(engine_path, "wb") as f:
#     f.write(engineString)
    
import pycuda.autoinit
import pycuda.driver as cuda
import os
import copy
import mmcv
import argparse
import tensorrt as trt
import numpy as np
from mmcv import Config
# from mmdeploy.backend.tensorrt import load_tensorrt_plugin
import sys
sys.path.append(".")
from convert import build_engine
from utils import HostDeviceMem, get_logger, create_engine_context, get_calibrator
from loaders import build_dataset, build_dataloader
import importlib

def parse_args():
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT")
    parser.add_argument("config", help="config file path")
    parser.add_argument("onnx")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--max_workspace_size", type=int, default=1)
    parser.add_argument(
        "--calibrator", type=str, default=None, help="[legacy, entropy, minmax]"
    )
    parser.add_argument(
        "--length", type=int, default=500, help="length of data to calibrate"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # load_tensorrt_plugin()

    config = Config.fromfile(args.config)
    
    # register custom module
    importlib.import_module('third_party')
    importlib.import_module('loaders')
    importlib.import_module('sparseocc_onnx_trt')

    dynamic_input = None
    if config.dynamic_input:
        dynamic_input = [
            config.dynamic_input.min,
            config.dynamic_input.reg,
            config.dynamic_input.max,
        ]

    output = os.path.split(args.onnx)[1].split(".")[0]
    if args.int8:
        if args.calibrator is not None:
            output += f"_{args.calibrator}"
        output += "_int8"
    if args.fp16:
        output += "_fp16"
    output = os.path.join(config.TENSORRT_PATH, output)

    dataset = build_dataset(cfg=config.data.quant)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=6, shuffle=False, dist=False
    )
    
    calibrator = None
    if args.calibrator is not None:
        TRT_LOGGER = get_logger(trt.Logger.INTERNAL_ERROR)
        engine_fp32_path = (
            os.path.split(args.onnx)[1]
            .replace(".onnx", ".trt")
            .replace("_cp2.", "_cp.")
            .replace("_cp2_", "_cp_")
        )
        engine_fp32_path = os.path.join(config.TENSORRT_PATH, engine_fp32_path)
        assert os.path.exists(engine_fp32_path), "Engine of FP32 should be built first."
        engine, context = create_engine_context(engine_fp32_path, TRT_LOGGER)
        stream = cuda.Stream()

        for key in config.default_shapes:
            if key in locals():
                raise RuntimeError(f"Variable {key} has been defined.")
            locals()[key] = config.default_shapes[key]
        batch_size = loader.batch_size

        output_shapes = {}
        for key in config.output_shapes.keys():
            shape = config.output_shapes[key][:]
            for shape_i in range(len(shape)):
                if isinstance(shape[shape_i], str):
                    shape[shape_i] = eval(shape[shape_i])
            output_shapes[key] = shape
        host_device_mem_dic = {}
        for name in output_shapes.keys():
            shape = output_shapes[name]
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, np.float32)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            host_device_mem_dic[name] = HostDeviceMem(name, host_mem, device_mem)

        class Calibrator(get_calibrator(args.calibrator)):
            def __init__(self, *args, **kwargs):
                super(Calibrator, self).__init__(*args, **kwargs)
                self.prev_bev_lst = []

            def decode_data(self, data):
                breakpoint()
                img = data["img"][0].data[0].numpy()
                img_metas = data["img_metas"][0].data[0]
                prev_bev = self.prev_bev_lst[self.current_batch].get("prev_bev", None)
                use_prev_bev = self.prev_bev_lst[self.current_batch].get(
                    "use_prev_bev", None
                )
                can_bus = self.prev_bev_lst[self.current_batch].get("can_bus", None)
                lidar2img = np.stack(img_metas[0]["lidar2img"], axis=0)

                for name in self.names:
                    if name == "image":
                        img = img.reshape(-1).astype(np.float32)
                        assert self.host_device_mem_dic[name].host.nbytes == img.nbytes
                        self.host_device_mem_dic[name].host = img
                    elif name == "prev_bev":
                        prev_bev = prev_bev.reshape(-1).astype(np.float32)
                        assert (
                            self.host_device_mem_dic[name].host.nbytes
                            == prev_bev.nbytes
                        )
                        self.host_device_mem_dic[name].host = prev_bev
                    elif name == "use_prev_bev":
                        use_prev_bev = use_prev_bev.reshape(-1).astype(np.float32)
                        assert (
                            self.host_device_mem_dic[name].host.nbytes
                            == use_prev_bev.nbytes
                        )
                        self.host_device_mem_dic[name].host = use_prev_bev
                    elif name == "can_bus":
                        can_bus = can_bus.reshape(-1).astype(np.float32)
                        assert (
                            self.host_device_mem_dic[name].host.nbytes == can_bus.nbytes
                        )
                        self.host_device_mem_dic[name].host = can_bus
                    elif name == "lidar2img":
                        lidar2img = lidar2img.reshape(-1).astype(np.float32)
                        assert (
                            self.host_device_mem_dic[name].host.nbytes
                            == lidar2img.nbytes
                        )
                        self.host_device_mem_dic[name].host = lidar2img
                    else:
                        raise RuntimeError(f"Cannot find input name {name}.")

        calibrator = Calibrator(config, loader, args.length)

        input_shapes = calibrator.input_shapes
        host_device_mem_dic.update(calibrator.host_device_mem_dic)

        names = list(engine)
        bindings = [int(host_device_mem_dic[name].device) for name in names]

        prev_bev = np.random.randn(config.bev_h_ * config.bev_w_, 1, config._dim_)
        prev_frame_info = {
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }
        
        prog_bar = mmcv.ProgressBar(calibrator.num_batch * calibrator.batch_size)
        for i, data in enumerate(loader):
            if i >= calibrator.num_batch:
                break
            img = data["img"][0].data[0].numpy()
            img_metas = data["img_metas"][0].data[0]
            use_prev_bev = np.array([1.0])
            if img_metas[0]["scene_token"] != prev_frame_info["scene_token"]:
                use_prev_bev = np.array([0.0])
            prev_frame_info["scene_token"] = img_metas[0]["scene_token"]
            tmp_pos = copy.deepcopy(img_metas[0]["can_bus"][:3])
            tmp_angle = copy.deepcopy(img_metas[0]["can_bus"][-1])
            if use_prev_bev[0] == 1:
                img_metas[0]["can_bus"][:3] -= prev_frame_info["prev_pos"]
                img_metas[0]["can_bus"][-1] -= prev_frame_info["prev_angle"]
            else:
                img_metas[0]["can_bus"][-1] = 0
                img_metas[0]["can_bus"][:3] = 0
            can_bus = img_metas[0]["can_bus"]
            lidar2img = np.stack(img_metas[0]["lidar2img"], axis=0)
            batch_size = len(img)

            for name in names:
                if name == "image":
                    img = img.reshape(-1).astype(np.float32)
                    assert host_device_mem_dic[name].host.nbytes == img.nbytes
                    host_device_mem_dic[name].host = img
                elif name == "prev_bev":
                    prev_bev = prev_bev.reshape(-1).astype(np.float32)
                    assert host_device_mem_dic[name].host.nbytes == prev_bev.nbytes
                    host_device_mem_dic[name].host = prev_bev
                elif name == "use_prev_bev":
                    use_prev_bev = use_prev_bev.reshape(-1).astype(np.float32)
                    assert host_device_mem_dic[name].host.nbytes == use_prev_bev.nbytes
                    host_device_mem_dic[name].host = use_prev_bev
                elif name == "can_bus":
                    can_bus = can_bus.reshape(-1).astype(np.float32)
                    assert host_device_mem_dic[name].host.nbytes == can_bus.nbytes
                    host_device_mem_dic[name].host = can_bus
                elif name == "lidar2img":
                    lidar2img = lidar2img.reshape(-1).astype(np.float32)
                    assert host_device_mem_dic[name].host.nbytes == lidar2img.nbytes
                    host_device_mem_dic[name].host = lidar2img

            calibrator.prev_bev_lst.append(
                {"use_prev_bev": use_prev_bev, "prev_bev": prev_bev, "can_bus": can_bus}
            )
            [
                cuda.memcpy_htod(
                    host_device_mem_dic[name].device, host_device_mem_dic[name].host
                )
                for name in names
            ]

            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

            [
                cuda.memcpy_dtoh(
                    host_device_mem_dic[name].host, host_device_mem_dic[name].device
                )
                for name in names
            ]

            prev_bev = host_device_mem_dic["bev_embed"].host

            for _ in range(batch_size):
                prog_bar.update()

    build_engine(
        args.onnx,
        output + ".trt",
        dynamic_input=dynamic_input,
        int8=args.int8,
        fp16=args.fp16,
        max_workspace_size=args.max_workspace_size,
        calibrator=calibrator,
    )


if __name__ == "__main__":
    main()
