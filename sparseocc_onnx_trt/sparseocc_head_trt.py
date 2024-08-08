from ..third_party.sparseocc.models import SparseOccHead

class SparseOccHeadTRT(SparseOccHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def merge_occ_pred(self, occ_preds, mask_preds, class_preds):
        mask_cls = class_preds[-1].sigmoid()
        mask_pred = mask_preds[-1].sigmoid()
        occ_loc = occ_preds[-1][0]
        
        sem_pred = self.merge_semseg(mask_cls, mask_pred)  # [B, C, N]
        
        # return tensors instead of dict
        if self.panoptic:
            pano_inst, pano_sem = self.merge_panoseg(mask_cls, mask_pred)  # [B, C, N]
            return sem_pred, occ_loc, pano_inst, pano_sem
        else:
            return sem_pred, occ_loc
    
    def forward(self, mlvl_feats, img_shape, lidar2img, ego2lidar):
        occ_preds, mask_preds, class_preds = self.transformer(mlvl_feats, img_shape, lidar2img, ego2lidar)
        
        # return {
        #     'occ_preds': occ_preds, 
        #     'mask_preds': mask_preds, 
        #     'class_preds': class_preds
        # }
        return occ_preds, mask_preds, class_preds