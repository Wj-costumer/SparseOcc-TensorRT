import numpy as np
import torch
from torch.onnx import OperatorExportTypes
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint


@torch.no_grad()
def pytorch2onnx(
    config,
    checkpoint,
    output_file,
    opset_version=17,
    verbose=False,
    cuda=True,
    inputs_data=None,
):

    model = build_detector(config.model, test_cfg=config.get("test_cfg", None))
    checkpoint = load_checkpoint(model, checkpoint, map_location="cpu")
    if cuda:
        model.to("cuda")
    else:
        model.to("cpu")

    onnx_shapes = config.default_shapes
    input_shapes = config.input_shapes
    output_shapes = config.output_shapes
    if 'dynamic_axes' in config:
        dynamic_axes = config.dynamic_axes
    else:
        dynamic_axes = None
        
    for key in onnx_shapes:
        if key in locals():
            raise RuntimeError(f"Variable {key} has been defined.")
        locals()[key] = onnx_shapes[key]

    torch.random.manual_seed(0)
    inputs = {}
    for key in input_shapes.keys():
        if inputs_data is not None and key in inputs_data:
            inputs[key] = inputs_data[key]
            if isinstance(inputs[key], np.ndarray):
                inputs[key] = torch.from_numpy(inputs[key])
            assert isinstance(inputs[key], torch.Tensor)
        else:
            for i in range(len(input_shapes[key])):
                if isinstance(input_shapes[key][i], str):
                    input_shapes[key][i] = eval(input_shapes[key][i])
            inputs[key] = torch.randn(*input_shapes[key])
        if cuda:
            inputs[key] = inputs[key].cuda()

    if config.use_filename:
        inputs["img_filenames"] = tuple(['data/nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg', 
                                         'data/nuscenes/samples/CAM_FRONT_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_RIGHT__1533151603520482.jpg', 
                                         'data/nuscenes/samples/CAM_FRONT_LEFT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_LEFT__1533151603504799.jpg', 
                                         'data/nuscenes/samples/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg', 
                                         'data/nuscenes/samples/CAM_BACK_LEFT/n008-2018-08-01-15-16-36-0400__CAM_BACK_LEFT__1533151603547405.jpg', 
                                         'data/nuscenes/samples/CAM_BACK_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_BACK_RIGHT__1533151603528113.jpg'] * 8)
    
    model.forward = model.forward_trt
    input_name = list(inputs.keys())
    output_name = list(output_shapes.keys())

    inputs = tuple(inputs.values())

    torch.onnx.export(
        model,
        inputs,
        output_file,
        input_names=input_name,
        output_names=output_name,
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=False,
        verbose=verbose,
        opset_version=opset_version,
        # dynamic_axes=dynamic_axes,
        operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
    )

    print(f"ONNX file has been saved in {output_file}")
