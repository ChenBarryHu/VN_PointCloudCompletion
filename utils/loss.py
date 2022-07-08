import torch
import math
import os
import sys
import torch.nn as nn
import torch.nn.functional as F

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(proj_dir, "extensions/ChamferDistancePytorch"))
from chamfer3D import dist_chamfer_3D
from fscore import fscore


def calc_cd(output, gt, calc_f1=False, return_raw=False, normalize=False, separate=False):
    # cham_loss = dist_chamfer_3D.chamfer_3DDist()
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, idx1, idx2 = cham_loss(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))

    if separate:
        res = [torch.cat([torch.sqrt(dist1).mean(1).unsqueeze(0), torch.sqrt(dist2).mean(1).unsqueeze(0)]),
               torch.cat([dist1.mean(1).unsqueeze(0),dist2.mean(1).unsqueeze(0)])]
    else:
        res = [cd_p, cd_t]
    if calc_f1:
        f1, _, _ = fscore(dist1, dist2, 0.0001)
        res.append(f1)
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])
    return res

def calc_dcd(x, gt, alpha=1000, n_lambda=1, return_raw=False, non_reg=False):
    x = x.float()
    gt = gt.float()
    batch_size, n_x, _ = x.shape
    batch_size, n_gt, _ = gt.shape
    assert x.shape[0] == gt.shape[0]

    if non_reg:
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    cd_p, cd_t, dist1, dist2, idx1, idx2 = calc_cd(x, gt, return_raw=True)
    # dist1 (batch_size, n_gt): a gt point finds its nearest neighbour x' in x;
    # idx1  (batch_size, n_gt): the idx of x' \in [0, n_x-1]
    # dist2 and idx2: vice versa
    exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)

    loss1 = []
    loss2 = []
    for b in range(batch_size):
        count1 = torch.bincount(idx1[b])
        weight1 = count1[idx1[b].long()].float().detach() ** n_lambda
        weight1 = (weight1 + 1e-6) ** (-1) * frac_21
        loss1.append((- exp_dist1[b] * weight1 + 1.).mean())

        count2 = torch.bincount(idx2[b])
        weight2 = count2[idx2[b].long()].float().detach() ** n_lambda
        weight2 = (weight2 + 1e-6) ** (-1) * frac_12
        loss2.append((- exp_dist2[b] * weight2 + 1.).mean())

    loss1 = torch.stack(loss1)
    loss2 = torch.stack(loss2)
    loss = (loss1 + loss2) / 2

    res = [loss, cd_p, cd_t]
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])

    return res
