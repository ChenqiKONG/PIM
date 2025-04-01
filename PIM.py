from __future__ import print_function, division
import torch
import torch.nn as nn
import numpy as np
from mmseg.models.backbones.swin_transformer import SwinTransformer
from mmseg.models.backbones.swin_transformer_mask_branch import SwinTransformer_mask
from mmseg.models.decode_heads.uper_head import UPerHead
from mmseg.models.feature_local_enhance.feature_local_pim import feature_local_conv
from mmseg.models.feature_fusion.feature_fusion import feature_fusion

device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
print('device: ', device)

class Seg_Swin(nn.Module):

    def __init__(self, num_classes=2):
        super(Seg_Swin, self).__init__()
        self.num_classes = num_classes
        self.norm_cfg = dict(type='BN', requires_grad=True)
        self.backbone_swin = SwinTransformer(pretrain_img_size=512,
                                 patch_size=4,
                                 in_chans=3,
                                 embed_dim=96,
                                 depths=[2, 2, 6, 2],
                                 num_heads=[3, 6, 12, 24],
                                 window_size=7,
                                 mlp_ratio=4.,
                                 qkv_bias=True,
                                 qk_scale=None,
                                 drop_rate=0.,
                                 attn_drop_rate=0.,
                                 drop_path_rate=0.3,
                                 norm_layer=nn.LayerNorm,
                                 ape=False,
                                 patch_norm=True,
                                 out_indices=(0, 1, 2, 3),
                                 frozen_stages=-1,
                                 use_checkpoint=False)

                        
        self.decode_head = UPerHead(
            in_channels=[96, 192, 384, 768],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            num_classes=2,
            norm_cfg=self.norm_cfg,
            channels=512,
            dropout_ratio=0.1,
            align_corners=False,
        )
        
        self.boundary_head = UPerHead(
            in_channels=[96, 192, 384, 768],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            num_classes=2,
            norm_cfg=self.norm_cfg,
            channels=512,
            dropout_ratio=0.1,
            align_corners=False,
        )

        self.local_conv_enhancement = feature_local_conv()

        self.backbone_recon = SwinTransformer_mask(pretrain_img_size=512,
                                 patch_size=4,
                                 in_chans=3,
                                 embed_dim=96,
                                 depths=[2, 2, 6, 2],
                                 num_heads=[3, 6, 12, 24],
                                 window_size=7,
                                 mlp_ratio=4.,
                                 qkv_bias=True,
                                 qk_scale=None,
                                 drop_rate=0.,
                                 attn_drop_rate=0.,
                                 drop_path_rate=0.3,
                                 norm_layer=nn.LayerNorm,
                                 ape=False,
                                 patch_norm=True,
                                 out_indices=(0, 1, 2, 3),
                                 frozen_stages=-1,
                                 use_checkpoint=False)

                        
        self.recon_head = UPerHead(
            in_channels=[96, 192, 384, 768],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            num_classes=3,
            norm_cfg=self.norm_cfg,
            channels=512,
            dropout_ratio=0.1,
            align_corners=False,
        )

        self.feature_fusion = feature_fusion()

    def forward(self, x):
        feature_dc = self.backbone_swin(x)
        feature_dc = self.local_conv_enhancement(feature_dc)
        feature_rc = self.backbone_recon(x)
        recon_img = self.recon_head(feature_rc)
        feature_fus = self.feature_fusion(feature_dc, feature_rc)
        out1 = self.decode_head(feature_fus)
        out2 = self.boundary_head(feature_fus)
        return out1, out2, recon_img

def PIM_model(pretrained = True, model_path = None):
    model = Seg_Swin()
    if pretrained:
        state_dicts = torch.load(model_path, map_location='cuda:0')
        pretrained_dicts = state_dicts['state_dict']
        model_dict = model.state_dict()
        updated_dicts = {}

        for k, v in pretrained_dicts.items():
            for k_m, v_m in model_dict.items():
                if k == k_m and v.size() == v_m.size():
                    print(k_m, v_m.size())
                    updated_dicts[k_m] = v
        
        for k_m, v_m in model_dict.items():
            if 'backbone_swin' in k_m:
                k = k_m.replace('backbone_swin', 'backbone')
                v = pretrained_dicts[k]
                if v_m.size() == v.size():
                    updated_dicts[k_m] = v
                    print(k_m, v.size())
        
        for k_m, v_m in model_dict.items():
            if 'backbone_recon' in k_m:
                k = k_m.replace('backbone_recon', 'backbone')
                v = pretrained_dicts[k]
                if v_m.size() == v.size():
                    updated_dicts[k_m] = v
                    print(k_m, v.size())

        for k_m, v_m in model_dict.items():
            if 'boundary_head' in k_m:
                k = k_m.replace('boundary_head', 'decode_head')
                v = pretrained_dicts[k]
                if v_m.size() == v.size():
                    updated_dicts[k_m] = v
                    print(k_m, v.size())
        
        for k_m, v_m in model_dict.items():
            if 'recon_head' in k_m:
                k = k_m.replace('recon_head', 'decode_head')
                v = pretrained_dicts[k]
                if v_m.size() == v.size():
                    updated_dicts[k_m] = v
                    print(k_m, v.size())

        model_dict.update(updated_dicts)
        model.load_state_dict(model_dict)
        print('Load Successfully!')
    return model

if __name__ == "__main__":
    model_path = "./pretrained_models/moby_upernet_swin_tiny_patch4_window7_512x512.pth"
    model = PIM_model(pretrained = True, model_path = model_path)
    device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
    model = model.to(device)
    # print(model)
    print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
    faces = torch.rand(size=(2, 3, 512, 512))
    faces = faces.to(device)
    segmap, boundary_map, recon_img = model(faces)
    print(segmap.size())
    print(boundary_map.size())
    print(recon_img.size())