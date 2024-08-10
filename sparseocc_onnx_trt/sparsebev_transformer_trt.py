from third_party.sparseocc.models import SparseBEVSampling, sampling_4d, make_sample_points_from_bbox
import torch

class SparseBEVSamplingTRT(SparseBEVSampling):
    def __init__(self, 
                 embed_dims=None,
                 num_frames=None,
                 num_points=None,
                 num_groups=None,
                 num_levels=None,
                 pc_range=None
                ):
        super().__init__(
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_points=num_points,
            num_groups=num_groups,
            num_levels=num_levels,
            pc_range=pc_range
            )
    
    def inner_forward(self, query_bbox, query_feat, mlvl_feats, img_shape, lidar2img):
        '''
        query_bbox: [B, Q, 10]
        query_feat: [B, Q, C]
        mlvl_feats: [B, C, H, W]
        img_shape: [B, N, 3]
        lidar2img: [B, N, 4, 4]
        '''
        B, Q = query_bbox.shape[:2]
        image_h, image_w, _ = img_shape[0][0]

        # sampling offset of all frames
        sampling_offset = self.sampling_offset(query_feat)
        sampling_offset = sampling_offset.view(B, Q, self.num_groups * self.num_points, 3)
        sampling_points = make_sample_points_from_bbox(query_bbox, sampling_offset, self.pc_range)  # [B, Q, GP, 3]
        sampling_points = sampling_points.reshape(B, Q, 1, self.num_groups, self.num_points, 3)
        sampling_points = sampling_points.expand(B, Q, self.num_frames, self.num_groups, self.num_points, 3)

        # warp sample points based on velocity
        # if query_bbox.shape[-1] > 8:
        #     time_diff = img_metas[0]['time_diff']  # [B, F]
        #     time_diff = time_diff[:, None, :, None]  # [B, 1, F, 1]
        #     vel = query_bbox[..., 8:].detach()  # [B, Q, 2]
        #     vel = vel[:, :, None, :]  # [B, Q, 1, 2]
        #     dist = vel * time_diff  # [B, Q, F, 2]
        #     dist = dist[:, :, :, None, None, :]  # [B, Q, F, 1, 1, 2]
        #     sampling_points = torch.cat([
        #         sampling_points[..., 0:2] - dist,
        #         sampling_points[..., 2:3]
        #     ], dim=-1)

        # scale weights
        scale_weights = self.scale_weights(query_feat).view(B, Q, self.num_groups, 1, self.num_points, self.num_levels)
        scale_weights = torch.softmax(scale_weights, dim=-1)
        scale_weights = scale_weights.expand(B, Q, self.num_groups, self.num_frames, self.num_points, self.num_levels)

        # sampling
        sampled_feats = sampling_4d(
            sampling_points,
            mlvl_feats,
            scale_weights,
            lidar2img,
            image_h, image_w
        )  # [B, Q, G, FP, C]

        return sampled_feats
    
    def forward(self, query_bbox, query_feat, mlvl_feats, img_shape, lidar2img):
        # if self.training and query_feat.requires_grad:
        #     return cp(self.inner_forward, query_bbox, query_feat, mlvl_feats, img_metas, use_reentrant=False)
        # else:
        #     return self.inner_forward(query_bbox, query_feat, mlvl_feats, img_metas)
        return self.inner_forward(query_bbox, query_feat, mlvl_feats, img_shape, lidar2img)
