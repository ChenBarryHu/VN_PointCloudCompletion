import torch
import torch.nn as nn
from models.vn_layers import *
from models.dgcnn import *
from models.pcn import *

class PCNNet(nn.Module):
    def __init__(self, config, enc_type="dgcnn_fps", dec_type="vn_foldingnet", enc_pretrained="/cluster/53/mrhu/VN_PCN/experiments/07-09_448_points_dgcnn_fps_000/models/model_best.pth", dec_pretrained=None):
        super().__init__()
        if enc_type == "dgcnn_fps":
            self.encoder = DGCNN_fps(config, latent_dim=1024, grid_size=4, only_coarse=config.only_coarse).to(config.device)
        else:
            raise Exception(f"encoder type {enc_type} not supported yet")

        if dec_type == "vn_foldingnet":
            self.decoder = VN_FoldingNet().to(config.device)
        elif dec_type == "foldingnet":
            self.decoder = FoldingNet().to(config.device)
        else:
            raise Exception(f"encoder type {enc_type} not supported yet")

        if enc_pretrained is not None:
            self.encoder.load_state_dict(torch.load(enc_pretrained))
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, input):
        coarse, feature_global = self.encoder(input)
        fine = self.decoder(coarse, feature_global)
        return coarse, fine