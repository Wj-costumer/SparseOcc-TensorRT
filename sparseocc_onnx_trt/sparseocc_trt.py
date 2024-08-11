from third_party.sparseocc.models.detectors import SparseOcc
from mmdet.models import DETECTORS
import torch
from .utils import pad_multiple
import numpy as np
from mmcv.runner.fp16_utils import cast_tensor_type
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

@DETECTORS.register_module()
class SparseOccTRT(SparseOcc):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def extract_feat(self, img, img_shape, ori_shape, pad_shape):
        """Extract features from images and points."""
        if len(img.shape) == 6:
            img = img.flatten(1, 2)  # [B, TN, C, H, W]

        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        img = img.float()

        if self.data_aug is not None:
            if 'img_color_aug' in self.data_aug and self.data_aug['img_color_aug'] and self.training:
                img = self.color_aug(img)

            if 'img_norm_cfg' in self.data_aug:
                img_norm_cfg = self.data_aug['img_norm_cfg']

                norm_mean = torch.tensor(img_norm_cfg['mean'], device=img.device)
                norm_std = torch.tensor(img_norm_cfg['std'], device=img.device)

                if img_norm_cfg['to_rgb']:
                    img = img[:, [2, 1, 0], :, :]  # BGR to RGB

                img = img - norm_mean.reshape(1, 3, 1, 1)
                img = img / norm_std.reshape(1, 3, 1, 1)

            for b in range(B):
                img_shape_ = torch.tensor([img.shape[2], img.shape[3], img.shape[1]], device=img.device)
                img_shape[b] = torch.stack([img_shape_ for _ in range(N)], dim=0)
                ori_shape[b] = torch.stack([img_shape_ for _ in range(N)], dim=0)

            if 'img_pad_cfg' in self.data_aug:
                img_pad_cfg = self.data_aug['img_pad_cfg']
                img, img_shape, pad_shape = pad_multiple(img, img_shape, pad_shape, size_divisor=img_pad_cfg['size_divisor'])
                H, W = img.shape[-2:]

        input_shape = img.shape[-2:]
        # update real input shape of each single img
        # for img_meta in img_metas:
        #     img_meta.update(input_shape=input_shape)

        img_feats = self.extract_img_feat(img)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped, img_shape, ori_shape, pad_shape

    
    def forward_trt(self, img, img_shape, ori_shape, pad_shape, lidar2img, ego2lidar, img_timestamp, img_filenames):
        if self.pts_bbox_head.panoptic:
            sem_pred, occ_loc, pano_inst, pano_sem = self.simple_test(img, img_filenames, img_shape, ori_shape, pad_shape, lidar2img, ego2lidar, img_timestamp)
            sem_pred = sem_pred.cpu().numpy().astype(np.uint8)[0]
            occ_loc = occ_loc.cpu().numpy().astype(np.uint8)[0]
            pano_inst = pano_inst.cpu().numpy().astype(np.int16)
            pano_sem = pano_sem.cpu().numpy().astype(np.uint8)
            return torch.from_numpy(sem_pred), torch.from_numpy(occ_loc), torch.from_numpy(pano_inst), torch.from_numpy(pano_sem)
        else:
            sem_pred, occ_loc = self.simple_test(img, img_filenames, img_shape, ori_shape, pad_shape, lidar2img, ego2lidar, img_timestamp)
            sem_pred = sem_pred
            occ_loc = occ_loc
            breakpoint()
            return sem_pred, occ_loc

    def simple_test_pts(self, x, img_shape, lidar2img, ego2lidar):
        occ_preds, mask_preds, class_preds = self.pts_bbox_head(x, img_shape, lidar2img, ego2lidar)
        if self.pts_bbox_head.panoptic:
            sem_pred, occ_loc, pano_inst, pano_sem = self.pts_bbox_head.merge_occ_pred(occ_preds, mask_preds, class_preds)
            return sem_pred, occ_loc, pano_inst, pano_sem
        else:
            sem_pred, occ_loc = self.pts_bbox_head.merge_occ_pred(occ_preds, mask_preds, class_preds)
            return sem_pred, occ_loc

    def simple_test(self, img, img_filenames, img_shape, ori_shape, pad_shape, lidar2img, ego2lidar, img_timestamp):
        '''
            img: torch.tensor([B, N, C, H, W])
            img_filenames: tuple(str_1, ..., str_n*t)
            img_shape: torch.tensor([B, N, 3])
            ori_shape: torch.tensor([B, N, 3])
            pad_shape: torch.tensor([B, N, 3])
            lidar2img: torch.tensor([B, N*T, 4, 4])
            ego2lidar: torch.tensor([B, N*T, 4, 4])
            img_timestamp: torch.tensor([B, N*T])
        '''
        self.fp16_enabled = False
        assert img.shape[0] == 1  # batch_size = 1

        B, N, C, H, W = img.shape
        img = img.reshape(B, N//6, 6, C, H, W)
        num_frames = len(img_filenames) // 6

        img_shape_ = (H, W, C)
        img_shape[0] = torch.tensor([img_shape_ for _ in range(len(img_filenames))], device=img.device)
        ori_shape[0] = torch.tensor([img_shape_ for _ in range(len(img_filenames))], device=img.device)
        pad_shape[0] = torch.tensor([img_shape_ for _ in range(len(img_filenames))], device=img.device)

        img_feats_list, img_metas_list = [], []

        # extract feature frame by frame
        for i in range(num_frames):
            img_indices = list(np.arange(i * 6, (i + 1) * 6))

            # img_metas_curr = [{}]
            # for k in img_metas[0].keys():
            #     if isinstance(img_metas[0][k], list):
            #         img_metas_curr[0][k] = [img_metas[0][k][i] for i in img_indices]
            filenames_curr = [img_filenames[i] for i in img_indices]
            img_shape_curr = torch.stack([img_shape[0][i] for i in img_indices], dim=0).unsqueeze(0)
            ori_shape_curr = torch.stack([ori_shape[0][i] for i in img_indices], dim=0).unsqueeze(0)
            pad_shape_curr = torch.stack([pad_shape[0][i] for i in img_indices], dim=0).unsqueeze(0)
            lidar2img_curr = torch.stack([lidar2img[0][i] for i in img_indices], dim=0).unsqueeze(0)
            ego2lidar_curr = torch.stack([ego2lidar[0][i] for i in img_indices], dim=0).unsqueeze(0) 
            img_timestamp_curr = torch.tensor([img_timestamp[0][i] for i in img_indices], device=img.device)
            
            if img_filenames[img_indices[0]] in self.memory:
                # found in memory
                img_feats_curr = self.memory[img_filenames[img_indices[0]]]
            else:
                # extract feature and put into memory
                img_feats_curr, img_shape_curr, ori_shape_curr, pad_shape_curr = self.extract_feat(img[:, i], img_shape_curr, ori_shape_curr, pad_shape_curr)
                self.memory[img_filenames[img_indices[0]]] = img_feats_curr
                self.queue.put(img_filenames[img_indices[0]])
                while self.queue.qsize() > 16:  # avoid OOM
                    pop_key = self.queue.get()
                    self.memory.pop(pop_key)

            img_feats_list.append(img_feats_curr)
            img_metas_list.append(tuple([filenames_curr, img_shape_curr, ori_shape_curr, pad_shape_curr, lidar2img_curr, ego2lidar_curr, img_timestamp_curr]))
        # reorganize
        feat_levels = len(img_feats_list[0])
        img_feats_reorganized = []
        for j in range(feat_levels):
            feat_l = torch.cat([img_feats_list[i][j] for i in range(len(img_feats_list))], dim=0)
            feat_l = feat_l.flatten(0, 1)[None, ...]
            img_feats_reorganized.append(feat_l)

        # img_metas_reorganized = img_metas_list[0]
        # for i in range(1, len(img_metas_list)):
        #     for k, v in img_metas_list[i][0].items():
        #         if isinstance(v, list):
        #             img_metas_reorganized[0][k].extend(v)

        img_filenames, img_shape, ori_shape, pad_shape, lidar2img, ego2lidar, img_timestamp = img_metas_list[0]
        
        for i in range(1, len(img_metas_list)):
            img_shape = torch.cat([img_shape, img_metas_list[i][1]], dim=0)
            lidar2img = torch.cat([lidar2img, img_metas_list[i][4]], dim=0)
            ego2lidar = torch.cat([ego2lidar, img_metas_list[i][5]], dim=0)
            
        img_feats = img_feats_reorganized
        # img_metas = img_metas_reorganized
        img_feats = cast_tensor_type(img_feats, torch.half, torch.float32)
        
        # run detector
        return self.simple_test_pts(img_feats, img_shape, lidar2img, ego2lidar)
