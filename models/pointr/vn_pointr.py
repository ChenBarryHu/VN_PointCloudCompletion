import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils
from timm.models.layers import DropPath,trunc_normal_

from models.pointr.utils.dgcnn_group import DGCNN_Grouper, VN_DGCNN_Grouper
from models.vn_layers import *
import numpy as np
from knn_cuda import KNN
knn = KNN(k=8, transpose_mode=False)

def fps(pc, num):
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
    sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return sub_pc

def get_knn_index(coor_q, coor_k=None):
    coor_k = coor_k if coor_k is not None else coor_q
    # coor: bs, 3, np
    batch_size, _, num_points = coor_q.size()
    num_points_k = coor_k.size(2)

    with torch.no_grad():
        _, idx = knn(coor_k, coor_q)  # bs k np
        idx_base = torch.arange(0, batch_size, device=coor_q.device).view(-1, 1, 1) * num_points_k
        idx = idx + idx_base
        idx = idx.view(-1)
    
    return idx  # bs*k*np

# def vn_get_graph_feature(x, k=16, idx=None, x_coord=None):
#         batch_size = x.size(0)
#         num_points = x.size(3)
#         x = x.view(batch_size, -1, num_points)
#         if idx is None:
#             if x_coord is None: # dynamic knn graph
#                 _,idx = knn(x, x)
#             else:          # fixed knn graph with input point coordinates
#                 _,idx = knn(x_coord, x_coord)
#         device = torch.device('cuda')

#         idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

#         idx = idx + idx_base

#         idx = idx.view(-1)
    
#         _, num_dims, _ = x.size()
#         num_dims = num_dims // 3

#         x = x.transpose(2, 1).contiguous()
#         feature = x.view(batch_size*num_points, -1)[idx, :]
#         feature = feature.view(batch_size, num_points, k, num_dims, 3) 
#         x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
        
#         feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()
    
#         return feature

def get_graph_feature(x, knn_index, x_q=None):

        #x: bs, np, c, knn_index: bs*k*np
        k = 8
        batch_size, num_points, num_dims = x.size()
        num_query = x_q.size(1) if x_q is not None else num_points
        feature = x.view(batch_size * num_points, num_dims)[knn_index, :]
        feature = feature.view(batch_size, k, num_query, num_dims)
        x = x_q if x_q is not None else x
        x = x.view(batch_size, 1, num_query, num_dims).expand(-1, k, -1, -1)
        feature = torch.cat((feature - x, x), dim=-1)
        return feature  # b k np c


def vn_get_graph_feature(x, knn_index, x_q=None):

        #x: bs, np, c, knn_index: bs*k*np
        k = 8
        batch_size, num_dims, _, num_points = x.size()
        x = x.permute(0,3,1,2).reshape(batch_size * num_points, -1)

        feature = x[knn_index, :]
        # feature = x.view(batch_size * num_points, num_dims)[knn_index, :]
        num_query = x_q.size(1) if x_q is not None else num_points
        num_dims = num_dims
        feature = feature.view(batch_size, k, num_query, num_dims, 3)
        x = x_q if x_q is not None else x
        x = x.view(batch_size, 1, num_query, num_dims, 3).repeat(1, k, 1, 1, 1)
        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 4, 2, 1).contiguous()

        return feature  # b c 3 np k


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_vnq = VNLinear(128,384)
        self.proj_vnk = VNLinear(128,384)
        self.proj_vnv = VNLinear(128,384)
        self.proj_vn = VNLinear(384,128)

    def forward(self, vn_x):
        B, C,_, N = vn_x.shape
        q = self.proj_vnq(vn_x).reshape(B,self.num_heads,3*C//self.num_heads,3,N).permute(0,1,4,2,3).reshape(B,self.num_heads,N,-1)
        k = self.proj_vnk(vn_x).reshape(B,self.num_heads,3*C//self.num_heads,3,N).permute(0,1,4,2,3).reshape(B,self.num_heads,N,-1)
        v = self.proj_vnv(vn_x).reshape(B,self.num_heads,3*C//self.num_heads,3,N).permute(0,1,4,2,3).reshape(B,self.num_heads,N,-1)
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, N, 3*C,3).permute(0,2,3,1)
        x = self.proj_vn(x)
        x = self.proj_drop(x)
        return x



class CrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, v):
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_q = None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = CrossAttention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map = nn.Linear(dim*2, dim)

        self.knn_map_cross = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map_cross = nn.Linear(dim*2, dim)

    def forward(self, q, v, self_knn_index=None, cross_knn_index=None):
        # q = q + self.drop_path(self.self_attn(self.norm1(q)))
        norm_q = self.norm1(q)
        q_1 = self.self_attn(norm_q)

        if self_knn_index is not None:
            knn_f = get_graph_feature(norm_q, self_knn_index)
            knn_f = self.knn_map(knn_f)
            knn_f = knn_f.max(dim=1, keepdim=False)[0]
            q_1 = torch.cat([q_1, knn_f], dim=-1)
            q_1 = self.merge_map(q_1)
        
        q = q + self.drop_path(q_1)

        norm_q = self.norm_q(q)
        norm_v = self.norm_v(v)
        q_2 = self.attn(norm_q, norm_v)

        if cross_knn_index is not None:
            knn_f = get_graph_feature(norm_v, cross_knn_index, norm_q)
            knn_f = self.knn_map_cross(knn_f)
            knn_f = knn_f.max(dim=1, keepdim=False)[0]
            q_2 = torch.cat([q_2, knn_f], dim=-1)
            q_2 = self.merge_map_cross(q_2)

        q = q + self.drop_path(q_2)

        # q = q + self.drop_path(self.attn(self.norm_q(q), self.norm_v(v)))
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        return q

class VN_DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_q = None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = CrossAttention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)


        self.merge_map = nn.Linear(dim*2, dim)
        self.conv1 = VNLinearLeakyReLU(256, 128)
        self.conv2 = VNLinear(256, 128)
        self.conv3 = VNLinearLeakyReLU(256, 128)
        self.conv4 = VNLinear(256, 128)
        self.conv5 = VNLinearLeakyReLU(128, 256, dim=4)
        self.conv6 = VNLinearLeakyReLU(256, 128, dim=4)

        self.pool1 = VNMaxPool(128)
        self.pool2 = VNMaxPool(128)


    def forward(self, q, v, self_knn_index=None, cross_knn_index=None):
        # q = q + self.drop_path(self.self_attn(self.norm1(q)))
        bs,num_queries,_=q.shape
        norm_q = self.norm1(q)
        q_1 = self.self_attn(norm_q)
        q_1 = q_1.transpose(2,1).view(bs,-1,3,num_queries)
        
        if self_knn_index is not None:
            knn_f = vn_get_graph_feature(norm_q, self_knn_index)
            # knn_f = self.knn_map(knn_f)
            # knn_f = knn_f.max(dim=1, keepdim=False)[0]
            knn_f = self.conv1(knn_f)
            knn_f = self.pool1(knn_f)
            q_1 = torch.cat([q_1, knn_f], dim=1)
            # q_1 = self.merge_map(q_1)
            q_1 = self.conv2(q_1).contiguous()
        
        q_1 = q_1.view(bs,-1,num_queries).transpose(1,2).contiguous()
        q = q + self.drop_path(q_1)

        norm_q = self.norm_q(q)
        norm_v = self.norm_v(v)
        q_2 = self.attn(norm_q, norm_v)

        # here
        q_2 = q_2.transpose(2,1).view(bs,-1,3,num_queries)
        if cross_knn_index is not None:
            knn_f = vn_get_graph_feature(norm_v, cross_knn_index, norm_q)
            # knn_f = self.knn_map_cross(knn_f)
            knn_f = self.conv3(knn_f)
            # knn_f = knn_f.max(dim=1, keepdim=False)[0]
            knn_f = self.pool2(knn_f)
            # q_2 = torch.cat([q_2, knn_f], dim=-1)
            q_2 = torch.cat([q_2, knn_f], dim=1)
            # q_2 = self.merge_map_cross(q_2)
            q_2 = self.conv4(q_2).contiguous()
        q_2 = q_2.view(bs,-1,num_queries).transpose(1,2).contiguous()
        q = q + self.drop_path(q_2)

        # the following part might help in the long term, will comment back and try
        # vn_q = q.transpose(2,1).view(bs,-1,3,num_queries)
        # q_3 = self.norm2(q).transpose(1,2).contiguous()
        # q_3 = q_3.view(bs,-1,3,num_queries)
        # q_3 = self.conv6(self.conv5(q_3))
        # vn_q = (vn_q + self.drop_path(q_3)).contiguous()
        # q = vn_q.view(bs,-1,num_queries).transpose(1,2).contiguous()
        
        return q

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map = nn.Linear(dim*2, dim)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, knn_index = None):
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        norm_x = self.norm1(x)
        x_1 = self.attn(norm_x)

        if knn_index is not None:
            knn_f = get_graph_feature(norm_x, knn_index)
            knn_f = self.knn_map(knn_f)
            knn_f = knn_f.max(dim=1, keepdim=False)[0]
            x_1 = torch.cat([x_1, knn_f], dim=-1)
            x_1 = self.merge_map(x_1)
        
        x = x + self.drop_path(x_1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VN_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = VNLayerNorm(dim // 3)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = VNLayerNorm(dim // 3)


        self.conv1 = VNLinearLeakyReLU(256, 128)
        self.conv2 = VNLinear(256, 128)
        self.conv3 = VNLinearLeakyReLU(128, 256, dim=4)
        self.conv4 = VNLinearLeakyReLU(256, 128, dim=4)
        self.pool1 = mean_pool

    def forward(self, x, knn_index = None):
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        bs,num_points,_ = x.shape
        vn_x = x.transpose(1,2).view(bs,-1,3,num_points)
        norm_x = self.norm1(vn_x)

        x_1 = self.attn(norm_x)

        # x_1 = x_1.transpose(2,1).view(bs,-1,3,num_points)
        if knn_index is not None:
            knn_f = vn_get_graph_feature(norm_x, knn_index)
            knn_f = self.conv1(knn_f)
            knn_f = self.pool1(knn_f)
            bs,_,_,num_points = knn_f.shape
            # knn_f = knn_f.view(bs,-1,num_points).transpose(2,1)
            x_1 = torch.cat([x_1, knn_f], dim=1)
            x_1 = self.conv2(x_1)
        vn_x = vn_x + self.drop_path(x_1)
        norm_vn_x = self.norm2(vn_x)
        # norm_vn_x = norm_vn_x.view(bs,-1,3,num_points)
        
        x_2 = self.conv4(self.conv3(norm_vn_x))
        # x_2 = self.norm2(x).transpose(1,2).contiguous()
        # x_2 = x_2.view(bs,-1,3,num_points)
        # x_2 = self.conv4(self.conv3(x_2))
        vn_x = (vn_x + self.drop_path(x_2)).contiguous()
        x = vn_x.view(bs,-1,num_points).transpose(1,2).contiguous()
        return x

class VN_PCTransformer(nn.Module):
    """ Vision Transformer with support for point cloud completion
    """
    def __init__(self, in_chans=3, embed_dim=768, depth=[6, 8], num_heads=4, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                        num_query = 224, knn_layer = -1, dgcnn='vn_dgcnn', trans='vn_trans', memory_profile=False, only_coarse=True):
        super().__init__()

        self.dgcnn = dgcnn
        self.trans = trans
        self.memory_profile = memory_profile
        self.only_coarse = only_coarse

        self.num_features = self.embed_dim = embed_dim
        
        self.knn_layer = knn_layer


        if self.dgcnn == 'vn_dgcnn':
            self.grouper = VN_DGCNN_Grouper()  # B 3 N to B C(3) N(128) and B C(128) N(128)
            self.input_proj = nn.Sequential(
                nn.Conv1d(384, embed_dim, 1), # (128, embed_dim,1))
                nn.BatchNorm1d(embed_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv1d(embed_dim, embed_dim, 1)
            )
            self.vn_pos_embed = nn.Sequential(
                VNLinearAndLeakyReLU(1,128,4),
                VNLinear(128,128)
            )
            self.fourth_vn_pos_embed = nn.Sequential(
                VNLinearAndLeakyReLU(2,128,4),
                VNLinear(128,128)
            )
            self.vn_input_proj = nn.Sequential(
                VNLinearLeakyReLU(128,128,4),
                VNLinear(128,128)
            )
            self.pos_embed = nn.Sequential(
                nn.Conv1d(in_chans, 128, 1),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv1d(128, embed_dim, 1)
            )
        else:
            self.grouper = DGCNN_Grouper()
            self.input_proj = nn.Sequential(
                nn.Conv1d(128, embed_dim, 1), # (128, embed_dim,1))
                nn.BatchNorm1d(embed_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv1d(embed_dim, embed_dim, 1)
            )

            self.pos_embed = nn.Sequential(
                nn.Conv1d(in_chans, 128, 1),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv1d(128, embed_dim, 1)
            )

        
        if self.trans == 'trans':
            self.encoder = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate)
                for i in range(depth[0])])
            self.decoder = nn.ModuleList([
                DecoderBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate)
                for i in range(depth[1])])
            self.increase_dim = nn.Sequential(
                nn.Conv1d(embed_dim, 1024, 1),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv1d(1024, 1024, 1)
            )
            self.coarse_pred = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 3 * num_query)
            )
            self.mlp_query = nn.Sequential(
                nn.Conv1d(1024+3, 1024, 1),
                # nn.BatchNorm1d(1024),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv1d(1024, 1024, 1),
                # nn.BatchNorm1d(1024),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv1d(1024, embed_dim, 1)
            )
        elif self.trans == 'vn_trans':
            self.encoder = nn.ModuleList([
                VN_Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate)
                for i in range(depth[0])])
            self.decoder = nn.ModuleList([
                VN_DecoderBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate)
                for i in range(depth[1])])
            self.vn_increase_dim = nn.Sequential(
                VNLinearAndLeakyReLU(128,1024,4),
                VNLinear(1024,1024)
            )
            self.vn_global_pool = VNMaxPool(in_channels=1024)
            self.vn_coarse_pred = nn.Sequential(
                VNLinear(1024,512),
                VNLeakyReLU(512),
                VNLinear(512,1024)
            )
            self.vn_mlp_query = nn.Sequential(
                VNLinearLeakyReLU(1025,1024,dim=4),
                VNLinearLeakyReLU(1024,1024,dim=4),
                VNLinear(1024,embed_dim//3)
            )
        else:
            raise TypeError()

        self.num_query = num_query
        self.apply(self._init_weights)

        # trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.cls_pos, std=.02)
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def pos_encoding_sin_wave(self, coor):
        # ref to https://arxiv.org/pdf/2003.08934v2.pdf
        D = 64 #
        # normal the coor into [-1, 1], batch wise
        normal_coor = 2 * ((coor - coor.min()) / (coor.max() - coor.min())) - 1 

        # define sin wave freq
        freqs = torch.arange(D, dtype=torch.float).cuda() 
        freqs = np.pi * (2**freqs)       

        freqs = freqs.view(*[1]*len(normal_coor.shape), -1) # 1 x 1 x 1 x D
        normal_coor = normal_coor.unsqueeze(-1) # B x 3 x N x 1
        k = normal_coor * freqs # B x 3 x N x D
        s = torch.sin(k) # B x 3 x N x D
        c = torch.cos(k) # B x 3 x N x D
        x = torch.cat([s,c], -1) # B x 3 x N x 2D
        pos = x.transpose(-1,-2).reshape(coor.shape[0], -1, coor.shape[-1]) # B 6D N
        # zero_pad = torch.zeros(x.size(0), 2, x.size(-1)).cuda()
        # pos = torch.cat([x, zero_pad], dim = 1)
        # pos = self.pos_embed_wave(x)
        return pos

    def forward(self, inpc):
        '''
            inpc : input incomplete point cloud with shape B N(2048) C(3)
        '''
        if self.memory_profile:
            a = torch.cuda.memory_allocated(0)
        # build point proxy
        bs = inpc.size(0)
        coor, f = self.grouper(inpc.transpose(1,2).contiguous())
        if self.memory_profile:
            b = torch.cuda.memory_allocated(0)
            print("2 - After DGCNN:", torch.cuda.memory_allocated(0))
            print("2 -     memory consumed by DGCNN:", b-a)
            a = b
        num_centers = f.shape[-1]
        
        if self.dgcnn == 'dgcnn':
            f = f.view(bs,-1,num_centers)
            x = self.input_proj(f).transpose(1,2)

        elif self.dgcnn == 'vn_dgcnn':
            f = self.vn_input_proj(f).contiguous()
            f = f.view(bs,-1,num_centers) 
            x = f.transpose(1,2)

        knn_index = get_knn_index(coor)
        # NOTE: try to use a sin wave  coor B 3 N, change the pos_embed input dim
        # pos = self.pos_encoding_sin_wave(coor).transpose(1,2)
        # There are three options to choose from: vn_pos, repeat_pos, pos
        
        # 1st option:
        # vn_pos =  self.vn_pos_embed(coor.unsqueeze(1)).reshape(bs, -1, num_centers)
        # vn_pos = vn_pos.transpose(2,1)

        # 2nd option:
        # repeat_pos = coor.unsqueeze(1).expand(-1,128,-1,-1).reshape(bs, -1, num_centers).transpose(2,1)
        
        # 3rd option:
        # pos =  self.pos_embed(coor).transpose(1,2)

        # 4th option: combine coor with input centers and run thru a VN
        input_centers = torch.mean(inpc,dim=1)

        repeat_input_centers = input_centers.unsqueeze(-1).expand(-1,-1,num_centers).unsqueeze(1)

        fourth_pos = torch.concat([coor.unsqueeze(1), repeat_input_centers], dim=1)
        fourth_pos =  self.fourth_vn_pos_embed(fourth_pos).reshape(bs, -1, num_centers).transpose(1,2)

        #since input_proj kills the RE, we simply transpose
        # x = self.input_proj(f).transpose(1,2)

        # cls_pos = self.cls_pos.expand(bs, -1, -1)
        # cls_token = self.cls_pos.expand(bs, -1, -1)
        # x = torch.cat([cls_token, x], dim=1)
        # pos = torch.cat([cls_pos, pos], dim=1)
        
        if self.memory_profile:
            b = torch.cuda.memory_allocated(0)
            print("2 - After DGCNN-post-processing:", torch.cuda.memory_allocated(0))
            print("2 -     memory consumed by DGCNN-post-processing:", b-a)
            a = b
        # encoder
        for i, blk in enumerate(self.encoder):
            if i < self.knn_layer:
                x = blk(x+fourth_pos, knn_index)   # B N C
            else:
                x = blk(x+fourth_pos)
        # build the query feature for decoder
        # global_feature  = x[:, 0] # B C
        if self.memory_profile:
            b = torch.cuda.memory_allocated(0)
            print("2 - After trans_encoder:", torch.cuda.memory_allocated(0))
            print("2 -     memory consumed by trans_encoder:", b-a)
            a = b

        ####################### start of predicting coarse points #########################
        # Originalway
        if self.trans == 'trans':
            global_feature = self.increase_dim(x.transpose(1,2)) # B 1024 N 
            global_feature = torch.max(global_feature, dim=-1)[0] # B 1024

            coarse_point_cloud = self.coarse_pred(global_feature).reshape(bs, -1, 3)  #  B M C(3)


        # VN way:
        elif self.trans == 'vn_trans':
            vn_x = x.transpose(1,2).contiguous()
            vn_x = vn_x.view(bs, -1, 3, num_centers)
            global_feature = self.vn_increase_dim(vn_x)
            global_feature = self.vn_global_pool(global_feature).unsqueeze(-1)
            coarse_point_cloud = self.vn_coarse_pred(global_feature)
            coarse_point_cloud = coarse_point_cloud.squeeze(-1)
        
        if self.memory_profile:
            b = torch.cuda.memory_allocated(0)
            print("2 - After coarse_prediction:", torch.cuda.memory_allocated(0))
            print("2 -     memory consumed by coarse_prediction:", b-a)
            a = b

        

        ####################### end of predicting coarse points #########################

        


        ################### start of pre-decoder processing ######################
        q = None
        if not self.only_coarse:
            new_knn_index = get_knn_index(coarse_point_cloud.transpose(1, 2).contiguous())
            cross_knn_index = get_knn_index(coor_k=coor, coor_q=coarse_point_cloud.transpose(1, 2).contiguous())

            # Original way
            if self.trans == 'trans':
                global_feature.unsqueeze(-1)
                global_feature = global_feature.view(bs, -1)
                query_feature = torch.cat([
                    global_feature.unsqueeze(1).expand(-1, self.num_query, -1), 
                    coarse_point_cloud], dim=-1) # B M C+3 
                q = self.mlp_query(query_feature.transpose(1,2)).transpose(1,2) # B M C 

            # VN way
            elif self.trans == 'vn_trans':
                global_feature = global_feature.expand(-1,-1,-1,self.num_query)
                vn_coarse_pc = coarse_point_cloud.transpose(1,2).unsqueeze(1)
                query_feature = torch.cat([global_feature, vn_coarse_pc], dim=1)
                q = self.vn_mlp_query(query_feature).reshape(bs,-1,self.num_query).transpose(1,2)


            ################### end of pre-decoder processing ########################
            # decoder
            for i, blk in enumerate(self.decoder):
                if i < self.knn_layer:
                    q = blk(q, x, new_knn_index, cross_knn_index)   # B M C
                else:
                    q = blk(q, x)
            
            if self.memory_profile:
                b = torch.cuda.memory_allocated(0)
                print("2 - After transformer:", torch.cuda.memory_allocated(0))
                print("2 -     memory consumed by transformer:", b-a)
                a = b

        inp_sparse = fps(inpc, 224)
        coarse_point_cloud = torch.cat([coarse_point_cloud, inp_sparse], dim=1).contiguous()
        return coarse_point_cloud, global_feature