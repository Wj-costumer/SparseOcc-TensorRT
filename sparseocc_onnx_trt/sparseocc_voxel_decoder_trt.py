from third_party.sparseocc.models import SparseVoxelDecoder, SparseVoxelDecoderLayer, index2point, point2bbox, upsample, encode_bbox
from .utils import DUMP, generate_grid, batch_indexing
from .sparsebev_transformer_trt import SparseBEVSamplingTRT
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseVoxelDecoderTRT(SparseVoxelDecoder):
    def __init__(self, 
                 embed_dims=None,
                 num_layers=None,
                 num_frames=None,
                 num_points=None,
                 num_groups=None,
                 num_levels=None,
                 num_classes=None,
                 semantic=False,
                 topk_training=None,
                 topk_testing=None,
                 pc_range=None):
        super().__init__(
            embed_dims=embed_dims,
            num_layers=num_layers,
            num_frames=num_frames,
            num_points=num_points,
            num_groups=num_groups,
            num_levels=num_levels,
            num_classes=num_classes,
            pc_range=pc_range,
            semantic=semantic,
            topk_training=topk_training,
            topk_testing=topk_testing
        )
        
        self.decoder_layers = nn.ModuleList()
        self.lift_feat_heads = nn.ModuleList()
        #self.occ_pred_heads = nn.ModuleList()
        
        if semantic:
            self.seg_pred_heads = nn.ModuleList()

        for i in range(num_layers):
            self.decoder_layers.append(SparseVoxelDecoderLayerTRT(
                 embed_dims=embed_dims,
                 num_frames=num_frames,
                 num_points=num_points // (2 ** i),
                 num_groups=num_groups,
                 num_levels=num_levels,
                 pc_range=pc_range,
                 self_attn=i in [0, 1]
            ))
            self.lift_feat_heads.append(nn.Sequential(
                nn.Linear(embed_dims, embed_dims * 8),
                nn.ReLU(inplace=True)
            ))
            #self.occ_pred_heads.append(nn.Linear(embed_dims, 1))

            if semantic:
                self.seg_pred_heads.append(nn.Linear(embed_dims, num_classes))
        
    def forward(self, mlvl_feats, img_shape, lidar2img):
        occ_preds = []
        
        topk = self.topk_training if self.training else self.topk_testing

        B = 1
        # init query coords
        interval = 2 ** self.num_layers
        query_coord = generate_grid(self.voxel_dim, interval).expand(B, -1, -1)  # [B, N, 3]
        query_feat = torch.zeros([B, query_coord.shape[1], self.embed_dims], device=query_coord.device)  # [B, N, C]

        for i, layer in enumerate(self.decoder_layers):
            DUMP.stage_count = i
            
            interval = 2 ** (self.num_layers - i)  # 8 4 2 1

            # bbox from coords
            query_bbox = index2point(query_coord, self.pc_range, voxel_size=0.4)  # [B, N, 3]
            query_bbox = point2bbox(query_bbox, box_size=0.4 * interval)  # [B, N, 6]
            query_bbox = encode_bbox(query_bbox, pc_range=self.pc_range)  # [B, N, 6]

            # transformer layer
            query_feat = layer(query_feat, query_bbox, mlvl_feats, img_shape, lidar2img)  # [B, N, C]
            
            # upsample 2x
            query_feat = self.lift_feat_heads[i](query_feat)  # [B, N, 8C]
            query_feat_2x, query_coord_2x = upsample(query_feat, query_coord, interval // 2)

            if self.semantic:
                seg_pred_2x = self.seg_pred_heads[i](query_feat_2x)  # [B, K, CLS]
            else:
                seg_pred_2x = None

            # sparsify after seg_pred
            non_free_prob = 1 - F.softmax(seg_pred_2x, dim=-1)[..., -1]  # [B, K]
            indices = torch.topk(non_free_prob, k=topk[i], dim=1)[1]  # [B, K]

            query_coord_2x = batch_indexing(query_coord_2x, indices, layout='channel_last')  # [B, K, 3]
            query_feat_2x = batch_indexing(query_feat_2x, indices, layout='channel_last')  # [B, K, C]
            seg_pred_2x = batch_indexing(seg_pred_2x, indices, layout='channel_last')  # [B, K, CLS]

            occ_preds.append((
                torch.div(query_coord_2x, interval // 2, rounding_mode='trunc').long(),
                None,
                seg_pred_2x,
                query_feat_2x,
                interval // 2)
            )

            query_coord = query_coord_2x.detach()
            query_feat = query_feat_2x.detach()

        return occ_preds

class SparseVoxelDecoderLayerTRT(SparseVoxelDecoderLayer):
    def __init__(self,
                 embed_dims=None,
                 num_frames=None,
                 num_points=None,
                 num_groups=None,
                 num_levels=None,
                 pc_range=None,
                 self_attn=True):
        super().__init__(
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_points=num_points,
            num_groups=num_groups,
            num_levels=num_levels,
            pc_range=pc_range,
            self_attn=self_attn
        )

        self.sampling = SparseBEVSamplingTRT(
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_groups=num_groups,
            num_points=num_points,
            num_levels=num_levels,
            pc_range=pc_range
        )
        
    def forward(self, query_feat, query_bbox, mlvl_feats, img_shape, lidar2img):
        query_pos = self.position_encoder(query_bbox[..., :3])
        query_feat = query_feat + query_pos

        if self.self_attn is not None:
            query_feat = self.norm1(self.self_attn(query_bbox, query_feat))
        sampled_feat = self.sampling(query_bbox, query_feat, mlvl_feats, img_shape, lidar2img)
        query_feat = self.norm2(self.mixing(sampled_feat, query_feat))
        query_feat = self.norm3(self.ffn(query_feat))

        return query_feat