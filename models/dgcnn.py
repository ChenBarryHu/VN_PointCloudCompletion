import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from models.utils.transform_net import Transform_Net
from models.utils.dgcnn_util import get_graph_feature

class DGCNN(nn.Module):
    def __init__(self, args, latent_dim=1024, grid_size=4, only_coarse=False):
        super(DGCNN, self).__init__()
        self.latent_dim = latent_dim
        self.args = args
        self.num_coarse = 448
        self.n_knn = 40
        self.transform_net = Transform_Net(args)
        self.grid_size = 4
        self.only_coarse = only_coarse
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, 1024, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
        #                            self.bn7,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
        #                            self.bn8,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.dp1 = nn.Dropout(p=0.5)
        # self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
        #                            self.bn9,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.dp2 = nn.Dropout(p=0.5)
        # self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
        #                            self.bn10,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv11 = nn.Conv1d(128, num_part, kernel_size=1, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )
        

    def forward(self, x):
        x = x.transpose(1,2)
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.n_knn)
        t = self.transform_net(x0)
        x = x.transpose(2, 1)
        x = torch.bmm(x, t)
        x = x.transpose(2, 1)

        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x)
        feature_global = x.max(dim=-1, keepdim=True)[0].squeeze(2)
        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, 3)

        if self.only_coarse:
            return coarse.contiguous(), None

        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)             # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)               # (B, 3, num_fine)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)                                           # (B, 2, num_fine)

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_dense)          # (B, 1024, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)                          # (B, 1024+2+3, num_fine)
    
        fine = self.final_conv(feat) + point_feat                                            # (B, 3, num_fine), fine point cloud

        # l = l.view(batch_size, -1, 1)
        # l = self.conv7(l)

        # x = torch.cat((x, l), dim=1)
        # x = x.repeat(1, 1, num_points)

        # x = torch.cat((x, x1, x2, x3), dim=1)

        # x = self.conv8(x)
        # x = self.dp1(x)
        # x = self.conv9(x)
        # x = self.dp2(x)
        # x = self.conv10(x)
        # x = self.conv11(x)
        
        # trans_feat = None
        return coarse.contiguous(), fine.transpose(1, 2).contiguous()