import os
import os.path as osp
import open3d as o3d
import numpy as np
import glob
from tqdm import tqdm
from pathlib import Path


def main():
    root = '/mnt_gx/usr/jiarun/data/training_data'

    for seq in ['zhongtai', 'nanhu', 'pingyao']:
    # for seq in ['pingyao']:
        file_names = glob.glob(osp.join(root, seq, 'pairs_data', 'submap', '*.pcd'))
        for file_name in tqdm(file_names):
            id_name = osp.basename(file_name).split('.')[0]
            new_file_name = osp.join(root, seq, 'pairs_data', 'downsampled_xyzi', id_name + '.npy')
            os.makedirs(osp.dirname(new_file_name), exist_ok=True)
            points = o3d.io.read_point_cloud(file_name).points

            # points = points[:, :3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            # pcd.colors = o3d.utility.Vector3dVector(intensity.reshape(num,1).repeat(3,axis=1))
            pcd = pcd.voxel_down_sample(0.4)
            points = np.array(pcd.points).astype(np.float32)
            np.save(new_file_name, points)


if __name__ == '__main__':
    main()
