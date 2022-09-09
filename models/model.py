import torch
import torch.nn as nn
import collections
from models.vn_layers import *
from models.dgcnn import *
from models.pcn import *
from models.pointr.vn_pointr import VN_PCTransformer

class PCNNet(nn.Module):
    def __init__(self, config, enc_type="dgcnn_fps", dec_type="foldingnet"):
        super().__init__()
        self.num_coarse = config.num_coarse
        self.only_coarse = config.only_coarse
        if enc_type == "dgcnn_fps":
            self.encoder = DGCNN_fps(config, latent_dim=1024, grid_size=4, only_coarse=config.only_coarse).to(config.device)
        elif enc_type == "vn_dgcnn_fps":
            self.encoder = VN_DGCNN_fps(config, only_coarse=config.only_coarse).to(config.device)
        elif enc_type == "vn_pointnet":
            self.encoder = VN_PointNet(config).to(config.device)
        elif enc_type == "vn_pointnet++":
            raise Exception(f"encoder type {enc_type} not supported yet")
        elif enc_type == "vn_pointr":
            self.encoder = VN_PCTransformer(in_chans = 3, embed_dim = 384, depth = [6, 8], drop_rate = 0., num_query = 224, knn_layer = 1, \
            dgcnn='vn_dgcnn', trans='vn_trans', memory_profile=False, only_coarse=True).to(config.device)
        else:
            raise Exception(f"encoder type {enc_type} not supported yet")

        
        if config.enc_pretrained != "none":
            dict = torch.load(config.enc_pretrained)
            # dict = collections.OrderedDict()
            # raw_dict = torch.load(config.enc_pretrained)['base_model']
            # for k, v in raw_dict.items():
            #     if 'module.base_model' in k:
            #         dict[k[18:]] = v
            
            self.encoder.load_state_dict(dict, strict=False)
            for param in self.encoder.parameters():
                param.requires_grad = False
        if not config.only_coarse:
            if dec_type == "vn_foldingnet":
                self.decoder = VN_FoldingNet(config).to(config.device)
            elif dec_type == "foldingnet":
                self.decoder = FoldingNet(config).to(config.device)
            elif dec_type == "attention_vn_foldingnet":
                self.decoder = Attention_VN_FoldingNet(config).to(config.device)
            else:
                raise Exception(f"encoder type {enc_type} not supported yet")

        

    def forward(self, input, rot=None):
        coarse, feature_global = self.encoder(input)
        
        if self.num_coarse == 448:
            if self.only_coarse:
                return coarse[1], None
            fine = self.decoder(coarse[0], feature_global, rot)
            return coarse[1], fine
        else:
            if self.only_coarse:
                return coarse, None
            fine = self.decoder(coarse, feature_global, rot)
            return coarse, fine