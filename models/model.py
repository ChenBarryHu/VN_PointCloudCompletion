import torch
import torch.nn as nn
import collections
from models.vn_layers import *
from models.dgcnn import *
from models.pcn import *

class PCNNet(nn.Module):
    def __init__(self, config, enc_type="dgcnn_fps", dec_type="foldingnet"):
        super().__init__()
        self.num_coarse = config.num_coarse
        self.only_coarse = config.only_coarse
        if enc_type == "dgcnn_fps":
            self.encoder = DGCNN_fps(config, latent_dim=1024, grid_size=4, only_coarse=config.only_coarse).to(config.device)
        else:
            raise Exception(f"encoder type {enc_type} not supported yet")

        if dec_type == "vn_foldingnet":
            self.decoder = VN_FoldingNet(config).to(config.device)
        elif dec_type == "foldingnet":
            self.decoder = FoldingNet(config).to(config.device)
        else:
            raise Exception(f"encoder type {enc_type} not supported yet")

        if config.enc_pretrained != "none":
            dict = collections.OrderedDict()
            raw_dict = torch.load(config.enc_pretrained)
            for k, v in raw_dict.items():
                if 'encoder' in k:
                    dict[k[8:]] = v
            
            self.encoder.load_state_dict(dict)
        for param in self.encoder.parameters():
            param.requires_grad = False

        

    def forward(self, input):
        coarse, feature_global = self.encoder(input)
        
        if self.num_coarse == 448:
            if self.only_coarse:
                return coarse[1], None
            fine = self.decoder(coarse[0], feature_global)
            return coarse[1], fine
        else:
            if self.only_coarse:
                return coarse, None
            fine = self.decoder(coarse, feature_global)
            return coarse, fine