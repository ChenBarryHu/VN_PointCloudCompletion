import torch
import torch.nn as nn
from models.vn_layers import *
from pointnet2_ops import pointnet2_utils
from models.transformer import VN_Block
def fps(pc, num):
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
    sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return sub_pc

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

        self.num_coarse = 1024 #self.num_dense // (self.grid_size ** 2)

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
            return coarse.contiguous(), feature_global
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)             # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)               # (B, 3, num_fine)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)                                           # (B, 2, num_fine)

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_dense)          # (B, 1024, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)                          # (B, 1024+2+3, num_fine)
    
        fine = self.final_conv(feat) + point_feat                                            # (B, 3, num_fine), fine point cloud

        return coarse.contiguous(), fine.transpose(1, 2).contiguous()
        
class VN_PointNet(nn.Module):
    """
    "PCN: Point Cloud Completion Network"
    (https://arxiv.org/pdf/1808.00671.pdf)

    Attributes:
        num_dense:  16384
        latent_dim: 1024
        grid_size:  4
        num_coarse: 1024
    """

    def __init__(self, config, num_dense=16384, latent_dim=1024):
        super().__init__()

        self.num_dense = num_dense
        self.latent_dim = latent_dim
        if config.num_coarse == 448:
            self.num_coarse = config.num_coarse // 2
        else:
            self.num_coarse = config.num_coarse

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

        if self.num_coarse == 224:
            inp_sparse = fps(xyz.contiguous(), 224)
            coarse_cat = torch.cat([coarse, inp_sparse], dim=1).contiguous()
            return (coarse.contiguous(), coarse_cat.contiguous()), feature_global

        return coarse.contiguous(), feature_global

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
            nn.Conv1d(2048 * 3 + 3 + 2, 512, 1),
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
        feature_global = feature_global.reshape(B, -1)
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)             # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)               # (B, 3, num_fine)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)                                           # (B, 2, num_fine)

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_dense)          # (B, 1024, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)                          # (B, 1024+2+3, num_fine)
    
        fine = self.final_conv(feat) + point_feat                                            # (B, 3, num_fine), fine point cloud

        return fine.transpose(1, 2).contiguous()

class VN_FoldingNet(nn.Module):
    def __init__(self, config,grid_size=4):
        super().__init__()

        self.grid_size = grid_size
        self.latent_dim = config.latent_dim
        self.num_dense=16384
        if config.num_coarse == 448:
            self.num_coarse=config.num_coarse // 2
            self.num_dense=14336
            self.grid_size=8
        else:
            self.num_coarse=config.num_coarse
            self.num_dense=19968
            self.grid_size=4

        self.final_conv = nn.Sequential(
            VNLinearLeakyReLU(1024+1+1, 256, dim=4),
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
        # self.final_conv_2 = nn.Sequential(
        #     VNLinearLeakyReLU(341+1, 256, dim=4),
        #     # nn.Conv1d(1024 + 3 + 2, 512, 1),
        #     # nn.BatchNorm1d(512),
        #     # nn.ReLU(inplace=True),
        #     VNLinearLeakyReLU(256,256, dim=4),
        #     # nn.Conv1d(512, 512, 1),
        #     # nn.BatchNorm1d(512),
        #     # nn.ReLU(inplace=True),
        #     VNLinear(256, 1)
        #     # nn.Conv1d(512, 3, 1)
        # )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        c = torch.zeros_like(a, dtype=torch.float)
        self.folding_seed = torch.cat([a, b, c], dim=0).reshape(1,1,3,-1).cuda()
        # self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)

    def forward(self, coarse, feature_global, rot=None):
        # print(f"Dimension of folding_seed: {self.folding_seed.shape}\n")
        if rot is not None:
            # print("rot is not none")
            folding_seed = self.folding_seed.squeeze(1).transpose(1,2)
            folding_seed = rot.transform_points(folding_seed).transpose(1,2).unsqueeze(1)
            # print(f"Dimension of rotated folding_seed: {self.folding_seed.shape}\n")
        else:
            folding_seed = self.folding_seed
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
    
        fine = self.final_conv(feat) + point_feat                                            # (B, 3, num_fine), fine point cloud

        return fine.squeeze(1).transpose(1, 2).contiguous()


class Attention_VN_FoldingNet(nn.Module):
    def __init__(self, config,grid_size=4):
        super().__init__()

        self.grid_size = grid_size
        self.latent_dim = config.latent_dim
        self.num_dense=16384
        if config.num_coarse == 448:
            self.num_coarse=config.num_coarse // 2
            self.num_dense=14336
            self.grid_size=8
        else:
            self.num_coarse=config.num_coarse
            self.num_dense=16384
            self.grid_size=4
        
        self.transformer = nn.ModuleList([
            VN_Block(
                dim=384, num_heads=8, mlp_ratio=1, qkv_bias=False, qk_scale=1,
                drop=0, attn_drop=0)
            for i in range(2)])

        self.final_conv = nn.Sequential(
            VNLinearLeakyReLU(self.latent_dim+1+1, 256, dim=4),
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
        # self.final_conv_2 = nn.Sequential(
        #     VNLinearLeakyReLU(341+1, 256, dim=4),
        #     # nn.Conv1d(1024 + 3 + 2, 512, 1),
        #     # nn.BatchNorm1d(512),
        #     # nn.ReLU(inplace=True),
        #     VNLinearLeakyReLU(256,256, dim=4),
        #     # nn.Conv1d(512, 512, 1),
        #     # nn.BatchNorm1d(512),
        #     # nn.ReLU(inplace=True),
        #     VNLinear(256, 1)
        #     # nn.Conv1d(512, 3, 1)
        # )
        self.downsize_global = VNLinear(2048, 384)
        # a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        # b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        # c = torch.zeros_like(a, dtype=torch.float)
        # self.folding_seed = torch.cat([a, b, c], dim=0).reshape(1,1,3,-1).cuda()

        a = torch.linspace(-1., 1., steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        c = torch.zeros_like(a, dtype=torch.float)
        self.folding_seed = torch.cat([a, b, c], dim=0).cuda()
        in_channel = 384
        hidden_dim = 256
        self.vn_folding1 = nn.Sequential(
            VNLinearLeakyReLU(in_channel+1, hidden_dim, dim=4),
            VNLinearLeakyReLU(hidden_dim, hidden_dim//2, dim=4),
            VNLinear(hidden_dim//2, 1)
        )

        self.vn_folding2 = nn.Sequential(
            VNLinearLeakyReLU(in_channel+1, hidden_dim, dim=4),
            VNLinearLeakyReLU(hidden_dim, hidden_dim//2, dim=4),
            VNLinear(hidden_dim//2, 1)
        )
        # self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)

    def forward(self, coarse, feature_global, rot=None):
        # first apply the transformer
        bs, N, _ = coarse.shape
        repeat_input_centers = coarse.unsqueeze(1).expand(-1,384,-1,-1).reshape(bs, -1, N).transpose(2,1)
        feature_global = self.downsize_global(feature_global).expand(-1,-1,-1,N).reshape(bs, -1, N).transpose(2,1)

        feature_global = feature_global + repeat_input_centers

        for i, blk in enumerate(self.transformer):
            feature_global = blk(feature_global)
            # if i < self.knn_layer:
            #     input_feature = blk(input_feature)   # B N C
            # else:
            #     input_feature = blk(input_feature)


        feature_global = feature_global.reshape(bs, N, -1, 3).permute(0,2,3,1)
        num_sample = self.grid_size * self.grid_size

        bs, dim, _,num_centers = feature_global.shape
        x = feature_global.permute(0,3,1,2).reshape(bs * num_centers,-1,3)
        features = x.view(bs * num_centers, dim,3,1).expand(-1, -1, -1,num_sample)
        seed = self.folding_seed.view(1, 1, 3, num_sample).expand(bs * num_centers,-1,-1,-1).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.vn_folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.vn_folding2(x)
        relative_xyz = fd2.reshape(bs, num_centers, 3, -1)
        rebuild_points = (relative_xyz + coarse.unsqueeze(-1)).transpose(2,3).reshape(bs, -1, 3)



        # print(f"Dimension of folding_seed: {self.folding_seed.shape}\n")
        # if rot is not None:
        #     # print("rot is not none")
        #     folding_seed = self.folding_seed.squeeze(1).transpose(1,2)
        #     folding_seed = rot.transform_points(folding_seed).transpose(1,2).unsqueeze(1)
        #     # print(f"Dimension of rotated folding_seed: {self.folding_seed.shape}\n")
        # else:
        #     folding_seed = self.folding_seed
        # B = coarse.shape[0]
        # point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)             # (B, num_coarse, S, 3)
        # point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1).unsqueeze(1)               # (B, 3, num_fine)

        # seed = folding_seed.unsqueeze(3).expand(B, -1, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
        # # print(f"Dimension of seed (after expansion): {seed.shape}\n")
        # seed = seed.reshape(B, -1, 3, self.num_dense)                                           # (B, 2, num_fine)
        # feat_global_dim = feature_global.shape[1]
        # # feature_global = feature_global[:,:(feat_global_dim//3)*3]
        # # feature_global = feature_global.expand(-1, -1, -1,self.num_dense)           # (B, 1024, num_fine)
        # # feature_global = feature_global.reshape(B,-1,3,self.num_dense)
        # feat = torch.cat([feature_global, seed, point_feat], dim=1)                          # (B, 1024+2+3, num_fine)
    
        # fine = self.final_conv(feat) + point_feat                                            # (B, 3, num_fine), fine point cloud

        # return fine.squeeze(1).transpose(1, 2).contiguous()
        return rebuild_points
