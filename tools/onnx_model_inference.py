import onnxruntime
import numpy as np
import cv2
import onnx 
import onnx_graphsurgeon as gs
import torch
import tensorrt as trt
print(trt.__version__)

if __name__ == '__main__':
    onnx_model_path = '/root/SparseOcc-TensorRT/checkpoints/onnx/sparseocc_r50_nuimg_704x256_8f_24e_v1_4.onnx'
    onnx_model = onnx.load(onnx_model_path)

    graph = gs.import_onnx(onnx_model)
    for node in graph.nodes:
        if "GridSample" in node.name:
            node.attrs = {"name": "GridSample3D", "version": 1, "namespace": ""}
            node.op = "GridSample3D"

    onnx.save(gs.export_onnx(graph), "./sparseocc_gs.onnx")
    
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    # ort_inputs = {'input': input_img, 'factor': input_factor}
    # ort_output = ort_session.run(None, ort_inputs)[0]
    
    try:
        onnx.checker.check_model(onnx_model)
        print("onnx model is ok")
    except:
        print("onnx model is wrong")
        
    # create builder and network
    logger = trt.Logger(trt.Logger.ERROR)
    breakpoint()
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    # config.set_memory_pool_limit = 1 << 20
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
    profile = builder.create_optimization_profile()

    profile.set_shape('input', [1,3 ,224 ,224], [1,3,224, 224], [1,3 ,224 ,224])
    # config.add_optimization_profile(profile)
    # create engine
    device = 0
    with torch.cuda.device(device):
        breakpoint()
        engine = builder.create_network(network, config)

    with open('model.engine', mode='wb') as f:
        f.write(bytearray(engine.serialize()))
        print("generating file done!")