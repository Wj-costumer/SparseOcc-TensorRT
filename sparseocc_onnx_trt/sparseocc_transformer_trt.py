from third_party.sparseocc.models import SparseOccTransformer, MaskFormerOccDecoder, MaskFormerOccDecoderLayer, MaskFormerSampling, sampling_4d, make_sample_points_from_mask
from .sparseocc_voxel_decoder_trt import SparseVoxelDecoderTRT
from mmdet.models.utils.builder import TRANSFORMER
import torch
import numpy as np
import copy
from .utils import DUMP


@TRANSFORMER.register_module()
class SparseOccTransformerTRT(SparseOccTransformer):
    def __init__(self, 
                 embed_dims,
                 num_layers,
                 num_queries,
                 num_frames,
                 num_points,
                 num_groups,
                 num_levels,
                 num_classes,
                 pc_range,
                 occ_size,
                 topk_training,
                 topk_testing,
                 **kwargs):
        super(SparseOccTransformerTRT, self).__init__(
            embed_dims, num_layers, num_queries, num_frames, num_points, num_groups,
            num_levels, num_classes, pc_range, occ_size, topk_training, topk_testing)
        
        self.voxel_decoder = SparseVoxelDecoderTRT(
            embed_dims=embed_dims,
            num_layers=3,
            num_frames=num_frames,
            num_points=num_points,
            num_groups=num_groups,
            num_levels=num_levels,
            num_classes=num_classes,
            pc_range=pc_range,
            semantic=True,
            topk_training=topk_training,
            topk_testing=topk_testing
        )
        self.decoder = MaskFormerOccDecoderTRT(
            embed_dims=embed_dims,
            num_layers=num_layers,
            num_frames=num_frames,
            num_queries=num_queries,
            num_points=num_points,
            num_groups=num_groups,
            num_levels=num_levels,
            num_classes=num_classes,
            pc_range=pc_range,
            occ_size=occ_size,
        )
        
    def forward(self, mlvl_feats, img_shape, lidar2img, ego2lidar):
        for lvl, feat in enumerate(mlvl_feats):
            B, TN, GC, H, W = feat.shape  # [B, TN, GC, H, W]
            N, T, G, C = 6, TN // 6, 4, GC // 4
            feat = feat.reshape(B, T, N, G, C, H, W)
            feat = feat.permute(0, 1, 3, 2, 5, 6, 4)  # [B, T, G, N, H, W, C]
            feat = feat.reshape(B*T*G, N, H, W, C)  # [BTG, N, H, W, C]
            mlvl_feats[lvl] = feat.contiguous()
        
        # lidar2img = np.asarray([m['lidar2img'] for m in img_metas]).astype(np.float32)
        # lidar2img = torch.from_numpy(lidar2img).to(feat.device)  # [B, N, 4, 4]
        # ego2lidar = np.asarray([m['ego2lidar'] for m in img_metas]).astype(np.float32)
        # ego2lidar = torch.from_numpy(ego2lidar).to(feat.device)  # [B, N, 4, 4]
        
        # img_metas = copy.deepcopy(img_metas)
        # img_metas[0]['lidar2img'] = torch.matmul(lidar2img, ego2lidar)
        lidar2img = torch.matmul(lidar2img, ego2lidar) # [B, N, 4, 4]
        occ_preds = self.voxel_decoder(mlvl_feats, img_shape, lidar2img)
        mask_preds, class_preds = self.decoder(occ_preds, mlvl_feats, img_shape, lidar2img)
        
        return occ_preds, mask_preds, class_preds
    
    
class MaskFormerOccDecoderTRT(MaskFormerOccDecoder):
    def __init__(self,
                 embed_dims=None,
                 num_layers=None,
                 num_frames=None,
                 num_queries=None,
                 num_points=None,
                 num_groups=None,
                 num_levels=None,
                 num_classes=None,
                 pc_range=None,
                 occ_size=None):
        super(MaskFormerOccDecoderTRT, self).__init__(
            embed_dims=embed_dims,
            num_layers=num_layers,
            num_queries=num_queries,
            num_frames=num_frames,
            num_points=num_points,
            num_groups=num_groups,
            num_levels=num_levels,
            num_classes=num_classes,
            pc_range=pc_range,
            occ_size=occ_size
        )

        self.decoder_layer = MaskFormerOccDecoderLayerTRT(
            embed_dims=embed_dims,
            mask_dim=embed_dims,
            num_frames=num_frames,
            num_points=num_points,
            num_groups=num_groups,
            num_levels=num_levels,
            num_classes=num_classes,
            pc_range=pc_range,
            occ_size=occ_size,
        )
        
    def forward(self, occ_preds, mlvl_feats, img_shape, lidar2img):
        occ_loc, occ_pred, _, mask_feat, _ = occ_preds[-1]
        bs = mask_feat.shape[0]
        query_feat = self.query_feat.weight[None].repeat(bs, 1, 1)
        query_pos = self.query_pos.weight[None].repeat(bs, 1, 1)
        
        valid_map, mask_pred, class_pred = self.decoder_layer.pred_segmentation(query_feat, mask_feat)
        
        class_preds = [class_pred]
        mask_preds = [mask_pred]

        for i in range(self.num_layers):
            DUMP.stage_count = i
            query_feat, valid_map, mask_pred, class_pred = self.decoder_layer(
                query_feat, valid_map, mask_pred, occ_preds, mlvl_feats, query_pos, img_shape, lidar2img
            )
            mask_preds.append(mask_pred)
            class_preds.append(class_pred)

        return mask_preds, class_preds


class MaskFormerOccDecoderLayerTRT(MaskFormerOccDecoderLayer):
    def __init__(self,
                 embed_dims=None,
                 mask_dim=None,
                 num_frames=None,
                 num_queries=None,
                 num_points=None,
                 num_groups=None,
                 num_levels=None,
                 num_classes=None,
                 pc_range=None,
                 occ_size=None):
        super().__init__(
            embed_dims=embed_dims,
            mask_dim=mask_dim,
            num_frames=num_frames,
            num_queries=num_queries,
            num_points=num_points,
            num_groups=num_groups,
            num_levels=num_levels,
            num_classes=num_classes,
            pc_range=pc_range,
            occ_size=occ_size
        )

        self.sampling = MaskFormerSamplingTRT(embed_dims, num_frames, num_groups, num_points, num_levels, pc_range=pc_range, occ_size=occ_size)
        
    def forward(self, query_feat, valid_map, mask_pred, occ_preds, mlvl_feats, query_pos, img_shape, lidar2img):
        """
        query_feat: [bs, num_query, embed_dim]
        valid_map: [bs, num_query, num_voxel]
        mask_pred: [bs, num_query, num_voxel]
        occ_preds: list(occ_loc, occ_pred, _, mask_feat, scale), all voxel decoder's outputs
            mask_feat: [bs, num_voxel, embed_dim]
            occ_pred: [bs, num_voxel]
            occ_loc: [bs, num_voxel, 3]
        """
        occ_loc, occ_pred, _, mask_feat, _ = occ_preds[-1]
        query_feat = self.norm1(self.self_attn(query_feat, query_pos=query_pos))

        # sampled_feat = self.sampling(query_feat, valid_map, occ_loc, mlvl_feats, img_shape, lidar2img)
        # query_feat = self.norm2(self.mixing(sampled_feat, query_feat))
        
        query_feat = self.norm3(self.ffn(query_feat))
        
        valid_map, mask_pred, class_pred = self.pred_segmentation(query_feat, mask_feat)
        return query_feat, valid_map, mask_pred, class_pred
    
    def pred_segmentation(self, query_feat, mask_feat):
        # if self.training and query_feat.requires_grad:
        #     return cp(self.inner_pred_segmentation, query_feat, mask_feat, use_reentrant=False)
        # else:
        #     return self.inner_pred_segmentation(query_feat, mask_feat)
        return self.inner_pred_segmentation(query_feat, mask_feat)
    
    def inner_pred_segmentation(self, query_feat, mask_feat):
        class_pred = self.classifier(query_feat)
        feat_proj = self.mask_proj(query_feat)
        mask_pred = torch.einsum("bqc,bnc->bqn", feat_proj, mask_feat)
        valid_map = (mask_pred > 0.0)
        
        return valid_map, mask_pred, class_pred
    
class MaskFormerSamplingTRT(MaskFormerSampling):
    def __init__(self, embed_dims=256, num_frames=4, num_groups=4, num_points=8, num_levels=4, pc_range=[], occ_size=[], init_cfg=None):
        super().__init__(
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_groups=num_groups,
            num_points=num_points,
            num_levels=num_levels,
            pc_range=pc_range,
            occ_size=occ_size,
            init_cfg=init_cfg
        )
        
    def inner_forward(self, query_feat, valid_map, occ_loc, mlvl_feats, img_shape, lidar2img):
        '''
        valid_map: [B, Q, W, H, D]
        query_feat: [B, Q, C]
        '''
        B, Q = query_feat.shape[:2]
        image_h, image_w, _ = img_shape[0][0]

        # sampling offset of all frames
        offset = self.offset(query_feat).view(B, Q, self.num_groups * self.num_points, 3)  # [B, Q, GP, 3]
        sampling_points = make_sample_points_from_mask(valid_map, self.pc_range, self.occ_size, self.num_groups*self.num_points, occ_loc, offset)
        sampling_points = sampling_points.reshape(B, Q, 1, self.num_groups, self.num_points, 3)
        sampling_points = sampling_points.expand(B, Q, self.num_frames, self.num_groups, self.num_points, 3)

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

    def forward(self, query_feat, valid_map, occ_loc,  mlvl_feats, img_shape, lidar2img):
        # if self.training and query_feat.requires_grad:
        #     return cp(self.inner_forward, query_feat, valid_map, occ_loc, mlvl_feats, img_metas, use_reentrant=False)
        # else:
        #     return self.inner_forward(query_feat, valid_map, occ_loc, mlvl_feats, img_metas)
        return self.inner_forward(query_feat, valid_map, occ_loc, mlvl_feats, img_shape, lidar2img)
