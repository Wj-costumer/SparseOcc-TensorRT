_base_ = ["r50_nuimg_704x256_8f.py"]

ONNX_PATH = "checkpoints/onnx"
TENSORRT_PATH = "checkpoints/tensorrt"

model = dict(
    type='SparseOccTRT',
    pts_bbox_head=dict(
        type='SparseOccHeadTRT',
        transformer=dict(
            type='SparseOccTransformerTRT'
        )
    )
)

data = dict(
    samples_per_gpu=1,
    quant=dict(
        type={{_base_.dataset_type}},
        data_root={{_base_.dataset_root}},
        occ_gt_root={{_base_.occ_gt_root}},
        ann_file={{_base_.data.train.ann_file}},
        pipeline={{_base_.test_pipeline}},
        classes={{_base_.det_class_names}},
        modality={{_base_.input_modality}},
        test_mode=True
    ),
)

# batch_size, num_classes, img_h, img_w
default_shapes = dict(
    batch_size=1,
    img_h=256,
    img_w=704,
    occ_size={{_base_._topk_testing_}}[-1],
    frames= {{_base_._num_frames_}},
    occ_cls=18,
    cameras=6,
)

input_shapes = dict(
    image=["batch_size", "cameras", 3, "img_h", "img_w"],
    img_shape=["batch_size", "cameras*frames", 3],
    ori_shape=["batch_size", "cameras*frames", 3],
    pad_shape=["batch_size", "cameras*frames", 3],
    lidar2img=["batch_size", "cameras*frames", 4, 4],
    ego2lidar=["batch_size", "cameras*frames", 4, 4],
    timestamp=["batch_size", "cameras*frames"]
)

output_shapes = dict(
    sem_pred=[1, 32000],
    occ_loc=[1, 32000, 3]
)

dynamic_input = None

use_filename=True