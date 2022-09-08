import torch
import torch.nn as nn
from models.vn_layers import *


def vn_get_graph_feature(x, knn_index, x_q=None):

        #x: bs, np, c, knn_index: bs*k*np
        k = 8
        batch_size, num_dims, _, num_points = x.size()
        x = x.permute(0,3,1,2).reshape(batch_size * num_points, -1)

        feature = x[knn_index, :]
        # feature = x.view(batch_size * num_points, num_dims)[knn_index, :]
        num_query = x_q.size(-1) if x_q is not None else num_points
        num_dims = num_dims
        feature = feature.view(batch_size, k, num_query, num_dims, 3)
        x = x_q if x_q is not None else x
        x = x.view(batch_size, 1, num_query, num_dims, 3).repeat(1, k, 1, 1, 1)
        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 4, 2, 1).contiguous()

        return feature  # b c 3 np k


class VN_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = VNLayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity() # DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = VNLayerNorm(dim)


        self.conv1 = VNLinearLeakyReLU(dim*2, dim)
        self.conv2 = VNLinear(dim*2, dim)
        self.conv3 = VNLinearLeakyReLU(dim, dim*2, dim=4)
        self.conv4 = VNLinearLeakyReLU(dim*2, dim, dim=4)
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
        self.proj_vnq = VNLinear(dim,dim)
        self.proj_vnk = VNLinear(dim,dim)
        self.proj_vnv = VNLinear(dim,dim)
        self.proj_vn = VNLinear(dim,dim)

    def forward(self, vn_x):
        B, C,_, N = vn_x.shape
        q = self.proj_vnq(vn_x).reshape(B,self.num_heads,C//self.num_heads,3,N).permute(0,1,4,2,3).reshape(B,self.num_heads,N,-1)
        k = self.proj_vnk(vn_x).reshape(B,self.num_heads,C//self.num_heads,3,N).permute(0,1,4,2,3).reshape(B,self.num_heads,N,-1)
        v = self.proj_vnv(vn_x).reshape(B,self.num_heads,C//self.num_heads,3,N).permute(0,1,4,2,3).reshape(B,self.num_heads,N,-1)
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, N, C,3).permute(0,2,3,1)
        x = self.proj_vn(x)
        x = self.proj_drop(x)
        return x