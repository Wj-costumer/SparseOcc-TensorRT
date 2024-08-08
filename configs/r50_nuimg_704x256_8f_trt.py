_base_ = ["r50_nuimg_704x256_8f.py"]

model = dict(
    type='SparseOccTRT',
    pts_bbox_head=dict(
        type='SparseOccHeadTRT',
        transformer=dict(
            type='SparseOccTransformerTRT'
        )
    )
)

# batch_size, num_classes, img_h, img_w
default_shapes = dict(
    batch_size=1,
    img_h=256,
    img_w=704,
    occ_size=_base_._topk_testing_[-1],
    frames= _base_._num_frames_,
    occ_cls=18,
    cameras=6,
)

input_shapes = dict(
    image=["batch_size", "cameras", 3, "img_h", "img_w"],
    img_shape=["batch_size", "cameras", 3],
    ori_shape=["batch_size", "cameras", 3],
    pad_shape=["batch_size", "cameras", 3],
    lidar2img=["batch_size", "cameras", 4, 4],
    ego2lidar=["batch_size", "cameras", 4, 4],
    timestamp=["batch_size", "cameras*frames"]
)

output_shapes = dict(
    sem_pred=["occ_size", "occ_cls"],
    occ_loc=["occ_size", 3]
)

use_filename=True