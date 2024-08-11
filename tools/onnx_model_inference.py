import onnxruntime
import numpy as np
import cv2
import onnx 
import torch
import tensorrt as trt

if __name__ == '__main__':
    onnx_model = '/root/SparseOcc-TensorRT/checkpoints/onnx/sparseocc_r50_nuimg_704x256_8f_24e_v1_2.onnx'
    # onnx_model = '/root/grid_sample.onnx'
    # ort_session = onnxruntime.InferenceSession(onnx_model)
    # ort_inputs = {'input': input_img, 'factor': input_factor}
    # ort_output = ort_session.run(None, ort_inputs)[0]
    onnx_model = onnx.load(onnx_model)
    breakpoint()
    # create builder and network
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # parse onnx
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    config = builder.create_builder_config()
    config.max_workspace_size = 1<<20
    profile = builder.create_optimization_profile()

    profile.set_shape('input', [1,3 ,224 ,224], [1,3,224, 224], [1,3 ,224 ,224])
    config.add_optimization_profile(profile)
    # create engine
    device = 'cuda: 0'
    with torch.cuda.device(device):
        engine = builder.build_engine(network, config)

    with open('model.engine', mode='wb') as f:
        f.write(bytearray(engine.serialize()))
        print("generating file done!")