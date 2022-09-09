import torch
from torch import nn
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from models.vn_layers import *
knn = KNN(k=16, transpose_mode=False)


class DGCNN_Grouper(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        K has to be 16
        '''
        self.input_trans = nn.Conv1d(3, 8, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 32),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 128),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

    
    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    @staticmethod
    def get_graph_feature(coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = 16
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = knn(coor_k, coor_q)  # bs k np
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, x):

        # x: bs, 3, np

        # bs 3 N(128)   bs C(224)128 N(128)
        coor = x
        f = self.input_trans(x)

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, 512) #[4,3,512], [4,32,512]
        f = self.get_graph_feature(coor_q, f_q, coor, f) #[4,64,512,16]
        f = self.layer2(f) #[4,64,512,16]
        f = f.max(dim=-1, keepdim=False)[0] #[4,64,512]
        coor = coor_q #[4,3,512]

        f = self.get_graph_feature(coor, f, coor, f) #[4,128,512,16]
        f = self.layer3(f) #[4,64,512,16]
        f = f.max(dim=-1, keepdim=False)[0] #[4,64,512]

        coor_q, f_q = self.fps_downsample(coor, f, 128) #[4,3,128], [4,64,128]
        f = self.get_graph_feature(coor_q, f_q, coor, f) #[4,128,128,16]
        f = self.layer4(f) #[4,128,128,16]
        f = f.max(dim=-1, keepdim=False)[0] #[4,128,128]
        coor = coor_q #[4,3,128]

        return coor, f


class VN_DGCNN_Grouper(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        K has to be 16
        '''

        # vn-based layers
        self.conv1 = nn.Sequential(
            VNLinearLeakyReLU(2, 32),
            # VNLinear(32,32)
        )
        # self.conv2 = VNLinearLeakyReLU(32, 32)
        # self.conv3 = VNLinearLeakyReLU(64, 32)
        self.conv4 = VNLinearLeakyReLU(64, 64)
        # self.conv4 = nn.Sequential(
        #     VNLinear(64, 64),
        #     VNBatchNorm(num_features=64,dim=5)
        # )
        self.conv5 = VNLinearLeakyReLU(128, 64)
        self.conv6 = VNLinearLeakyReLU(128, 128)

        self.pool1 = mean_pool
        self.pool2 = mean_pool
        self.pool3 = mean_pool
        self.pool4 = mean_pool

    
    @staticmethod
    def fps_downsample(coor, x, num_group):
        batch_size = coor.size(0)
        num_points = coor.size(2)
        x = x.view(batch_size, -1, num_points)

        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:,3:].view(batch_size, -1, 3, num_group)

        return new_coor, new_x

    @staticmethod
    def get_graph_feature(coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = 16
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = knn(coor_k, coor_q)  # bs k np
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)

        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    @staticmethod
    def vn_get_graph_feature(x, k=16, idx=None, x_coord=None):
        batch_size = x.size(0)
        num_points = x.size(3)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            if x_coord is None: # dynamic knn graph
                _,idx = knn(x, x)
            else:          # fixed knn graph with input point coordinates
                _,idx = knn(x_coord, x_coord)
        device = torch.device('cuda')
        idx = idx.transpose(2, 1).contiguous()
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

        idx = idx + idx_base

        idx = idx.view(-1)
    
        _, num_dims, _ = x.size()
        num_dims = num_dims // 3

        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size*num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims, 3) 
        x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
        
        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()
    
        return feature

    def forward(self, x):

        # x: bs, 3, np

        # bs 3 N(128)   bs C(224)128 N(128)
        coor = x
        x = x.unsqueeze(1)
        # f = self.input_trans(x)
        x = self.vn_get_graph_feature(x) # x = self.vn_get_graph_feature(x, x_coord=coor)
        x = self.conv1(x)
        # x = self.conv2(x)
        x1 = self.pool1(x)

        coor_q, f_q = self.fps_downsample(coor, x1, 512)
        f = self.vn_get_graph_feature(f_q) # f = self.vn_get_graph_feature(f_q, x_coord=coor_q)
        # f = self.conv3(f)
        f = self.conv4(f)
        f = self.pool2(f)

        f = self.vn_get_graph_feature(f) # f = self.vn_get_graph_feature(f, x_coord=coor_q)
        f = self.conv5(f)
        f = self.pool3(f)
        

        coor_q, f_q = self.fps_downsample(coor_q, f, 128)
        f = self.vn_get_graph_feature(f_q) # f = self.vn_get_graph_feature(f_q, x_coord=coor_q)
        f = self.conv6(f)
        f = self.pool4(f)

        coor = coor_q

        return coor, f