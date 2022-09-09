
from pyntcloud import PyntCloud
import numpy as np
import pandas as pd

def iou(preds, gt):
    # preds_occupy = preds[:, 1, :, :] >= thresh
    diff = np.sum(np.logical_xor(preds, gt))
    intersection = np.sum(np.logical_and(preds, gt))
    union = np.sum(np.logical_or(preds, gt))
    # num_fp = np.sum(np.logical_and(preds_occupy, gt[:, 0, :, :]))  # false positive
    # num_fn = np.sum(np.logical_and(np.logical_not(preds_occupy), gt[:, 1, :, :]))  # false negative
    # return np.array([diff, intersection, union, num_fp, num_fn])
    return intersection / union

def evaluate_iou(preds_pc, gt_pc, size_grid=64):
    pred_voxel = points_to_voxels(preds_pc, size_grid=size_grid)
    gt_voxel = points_to_voxels(gt_pc, size_grid=size_grid)
    return iou(pred_voxel, gt_voxel)


def voxel2mesh(voxels, surface_view):
    cube_verts = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0],
                  [1, 1, 1]]  # 8 points

    cube_faces = [[0, 1, 2], [1, 3, 2], [2, 3, 6], [3, 7, 6], [0, 2, 6], [0, 6, 4], [0, 5, 1],
                  [0, 4, 5], [6, 7, 5], [6, 5, 4], [1, 7, 3], [1, 5, 7]]  # 12 face

    cube_verts = np.array(cube_verts)
    cube_faces = np.array(cube_faces) + 1

    scale = 0.01
    cube_dist_scale = 1.1
    verts = []
    faces = []
    curr_vert = 0

    positions = np.where(voxels > 0.3)
    voxels[positions] = 1 
    for i, j, k in zip(*positions):
        # identifies if current voxel has an exposed face 
        if not surface_view or np.sum(voxels[i-1:i+2, j-1:j+2, k-1:k+2]) < 27:
            verts.extend(scale * (cube_verts + cube_dist_scale * np.array([[i, j, k]])))
            faces.extend(cube_faces + curr_vert)
            curr_vert += len(cube_verts)  
              
    return np.array(verts), np.array(faces)


def write_obj(filename, verts, faces):
    """ write the verts and faces on file."""
    with open(filename, 'w') as f:
        # write vertices
        f.write('g\n# %d vertex\n' % len(verts))
        for vert in verts:
            f.write('v %f %f %f\n' % tuple(vert))

        # write faces
        f.write('# %d faces\n' % len(faces))
        for face in faces:
            f.write('f %d %d %d\n' % tuple(face))


def voxel2obj(filename, pred, surface_view = True):
    verts, faces = voxel2mesh(pred, surface_view)
    write_obj(filename, verts, faces)

def ply_to_voxels(filename=None, size_grid=64):
    if filename is None:
        return None
    cloud = PyntCloud.from_file(filename)

    voxelgrid_id = cloud.add_structure("voxelgrid", n_x=size_grid, n_y=size_grid, n_z=size_grid)
    voxelgrid = cloud.structures[voxelgrid_id]
    # voxelgrid.plot(d=3, mode="density", cmap="hsv")

    x_cords = voxelgrid.voxel_x
    y_cords = voxelgrid.voxel_y
    z_cords = voxelgrid.voxel_z

    voxels = np.zeros((size_grid, size_grid, size_grid)).astype(np.bool)

    for x, y, z in zip(x_cords, y_cords, z_cords):
        voxels[x][y][z] = True
    
    return voxels


def points_to_voxels(points, size_grid=64):
    cloud = PyntCloud(pd.DataFrame(points, columns=["x","y","z"]))

    voxelgrid_id = cloud.add_structure("voxelgrid", n_x=size_grid, n_y=size_grid, n_z=size_grid)
    voxelgrid = cloud.structures[voxelgrid_id]
    # voxelgrid.plot(d=3, mode="density", cmap="hsv")

    x_cords = voxelgrid.voxel_x
    y_cords = voxelgrid.voxel_y
    z_cords = voxelgrid.voxel_z

    voxels = np.zeros((size_grid, size_grid, size_grid)).astype(np.bool)

    for x, y, z in zip(x_cords, y_cords, z_cords):
        voxels[x][y][z] = True
    
    return voxels