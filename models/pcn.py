import torch
import torch.nn as nn
from models.vn_layers import *

class VN_PCN(nn.Module):
    """
    "PCN: Point Cloud Completion Network"
    (https://arxiv.org/pdf/1808.00671.pdf)

    Attributes:
        num_dense:  16384
        latent_dim: 1024
        grid_size:  4
        num_coarse: 1024
    """

    def __init__(self, num_dense=16384, latent_dim=1024, grid_size=4, only_coarse=False):
        super().__init__()

        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.only_coarse = only_coarse

        assert self.num_dense % self.grid_size ** 2 == 0

        self.num_coarse = 448 #self.num_dense // (self.grid_size ** 2)

        self.first_conv = nn.Sequential(
            VNLinearLeakyReLU(1,128,dim=4), 
            #nn.Conv1d(3, 128, 1),
            # nn.BatchNorm1d(128),
            # nn.ReLU(inplace=True),
            VNLinear(128,512)
            # nn.Conv1d(128, 256, 1)
        )

        self.maxpool1 = VNMaxPool(512)

        self.second_conv = nn.Sequential(
            VNLinearLeakyReLU(1024,1024,dim=4), 
            # nn.Conv1d(512, 512, 1),
            # nn.BatchNorm1d(512),
            # nn.ReLU(inplace=True),
            VNLinear(1024,self.latent_dim * 2)
            # nn.Conv1d(512, self.latent_dim, 1)
        )
        self.maxpool2 = VNMaxPool(self.latent_dim * 2)

        self.mlp = nn.Sequential(
            VNLinearAndLeakyReLU(self.latent_dim * 2, 1024 * 2, dim=4, use_batchnorm='none'),
            # nn.Linear(self.latent_dim, 1024),
            # nn.ReLU(inplace=True),
            VNLinearAndLeakyReLU(1024*2, 1024, dim=4, use_batchnorm='none'),
            # nn.Linear(1024, 1024),
            # nn.ReLU(inplace=True),
            VNLinear(1024, self.num_coarse)
            # nn.Linear(1024, 3 * self.num_coarse)
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(1024 + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)

    def forward(self, xyz):
        B, N, _ = xyz.shape
        
        # encoder
        feature = self.first_conv(xyz.transpose(2, 1).unsqueeze(1))                                       # (B,  256, N)
        feature_global = self.maxpool1(feature).unsqueeze(-1)

        feature = torch.cat([feature_global.expand(-1, -1, -1, N), feature], dim=1)              # (B,  512, N)
        feature = self.second_conv(feature)                                                  # (B, 1024, N)
        feature_global = self.maxpool2(feature).unsqueeze(-1)                        # (B, 1024)
        
        # decoder
        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, 3)                    # (B, num_coarse, 3), coarse point cloud

        if self.only_coarse:
            return coarse.contiguous(), None
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)             # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)               # (B, 3, num_fine)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)                                           # (B, 2, num_fine)

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_dense)          # (B, 1024, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)                          # (B, 1024+2+3, num_fine)
    
        fine = self.final_conv(feat) + point_feat                                            # (B, 3, num_fine), fine point cloud

        return coarse.contiguous(), fine.transpose(1, 2).contiguous()
        

class PCN(nn.Module):
    """
    "PCN: Point Cloud Completion Network"
    (https://arxiv.org/pdf/1808.00671.pdf)

    Attributes:
        num_dense:  16384
        latent_dim: 1024
        grid_size:  4
        num_coarse: 1024
    """

    def __init__(self, num_dense=16384, latent_dim=1024, grid_size=4, only_coarse=False):
        super().__init__()

        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.only_coarse = only_coarse

        assert self.num_dense % self.grid_size ** 2 == 0

        self.num_coarse = self.num_dense // (self.grid_size ** 2)

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.latent_dim, 1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(1024 + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)

    def forward(self, xyz):
        B, N, _ = xyz.shape
        
        # encoder
        feature = self.first_conv(xyz.transpose(2, 1))                                       # (B,  256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]                          # (B,  256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)              # (B,  512, N)
        feature = self.second_conv(feature)                                                  # (B, 1024, N)
        feature_global = torch.max(feature,dim=2,keepdim=False)[0]                           # (B, 1024)
        
        # decoder
        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, 3)                    # (B, num_coarse, 3), coarse point cloud

        if self.only_coarse:
            return coarse.contiguous(), None
             
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)             # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)               # (B, 3, num_fine)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)                                           # (B, 2, num_fine)

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_dense)          # (B, 1024, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)                          # (B, 1024+2+3, num_fine)
    
        fine = self.final_conv(feat) + point_feat                                            # (B, 3, num_fine), fine point cloud

        return coarse.contiguous(), fine.transpose(1, 2).contiguous()

class FoldingNet(nn.Module):
    def __init__(self, config, grid_size=4):
        super().__init__()

        self.grid_size = grid_size
        
        if config.num_coarse == 448:
            self.num_coarse=config.num_coarse // 2
            self.num_dense=14336
            self.grid_size=8
        else:
            self.num_coarse=config.num_coarse
            self.num_dense=16384
            self.grid_size=4
        self.final_conv = nn.Sequential(
            nn.Conv1d(1024 + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)

    def forward(self, coarse, feature_global, rot=None):
        B = coarse.shape[0]
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)             # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)               # (B, 3, num_fine)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)                                           # (B, 2, num_fine)

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_dense)          # (B, 1024, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)                          # (B, 1024+2+3, num_fine)
    
        fine = self.final_conv(feat) + point_feat                                            # (B, 3, num_fine), fine point cloud

        return fine.transpose(1, 2).contiguous()

class VN_FoldingNet(nn.Module):
    def __init__(self, config, latent_dim=1024//3,grid_size=4):
        super().__init__()

        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.num_dense=16384
        if config.num_coarse == 448:
            self.num_coarse=config.num_coarse // 2
            self.num_dense=14336
            self.grid_size=8
        else:
            self.num_coarse=config.num_coarse
            self.num_dense=16384
            self.grid_size=4

        self.final_conv = nn.Sequential(
            VNLinearLeakyReLU(341+1+1, 256, dim=4),
            # nn.Conv1d(1024 + 3 + 2, 512, 1),
            # nn.BatchNorm1d(512),
            # nn.ReLU(inplace=True),
            VNLinearLeakyReLU(256,256, dim=4),
            # nn.Conv1d(512, 512, 1),
            # nn.BatchNorm1d(512),
            # nn.ReLU(inplace=True),
            VNLinear(256, 1)
            # nn.Conv1d(512, 3, 1)
        )
        self.final_conv_2 = nn.Sequential(
            VNLinearLeakyReLU(341+1, 256, dim=4),
            # nn.Conv1d(1024 + 3 + 2, 512, 1),
            # nn.BatchNorm1d(512),
            # nn.ReLU(inplace=True),
            VNLinearLeakyReLU(256,256, dim=4),
            # nn.Conv1d(512, 512, 1),
            # nn.BatchNorm1d(512),
            # nn.ReLU(inplace=True),
            VNLinear(256, 1)
            # nn.Conv1d(512, 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        c = torch.zeros_like(a, dtype=torch.float)
        self.folding_seed = torch.cat([a, b, c], dim=0).reshape(1,1,3,-1).cuda()
        # self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)

    def forward(self, coarse, feature_global, rot=None):
        # print(f"Dimension of folding_seed: {self.folding_seed.shape}\n")
        if rot is not None:
            folding_seed = self.folding_seed.squeeze(1).transpose(1,2)
            folding_seed = rot.transform_points(folding_seed).transpose(1,2).unsqueeze(1)
            # print(f"Dimension of rotated folding_seed: {self.folding_seed.shape}\n")
        B = coarse.shape[0]
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)             # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1).unsqueeze(1)               # (B, 3, num_fine)

        seed = folding_seed.unsqueeze(3).expand(B, -1, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
        # print(f"Dimension of seed (after expansion): {seed.shape}\n")
        seed = seed.reshape(B, -1, 3, self.num_dense)                                           # (B, 2, num_fine)
        feat_global_dim = feature_global.shape[1]
        feature_global = feature_global[:,:(feat_global_dim//3)*3]
        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_dense)          # (B, 1024, num_fine)
        feature_global = feature_global.reshape(B,-1,3,self.num_dense)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)                          # (B, 1024+2+3, num_fine)

        feat = self.final_conv(feat)
        feat = torch.cat([feature_global, feat], dim=1)
    
        fine = self.final_conv_2(feat) + point_feat                                            # (B, 3, num_fine), fine point cloud

        return fine.squeeze(1).transpose(1, 2).contiguous()
