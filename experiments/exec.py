import os.path as osp
import numpy as np
import torch
import argparse
from geotransformer.engine import SingleTester
from geotransformer.utils.torch import release_cuda
from model_infer import create_model
from config import make_cfg
from geotransformer.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)

def move_tensors_to_cuda(data_dict):
    # 检查 CUDA 是否可用
    if torch.cuda.is_available():
        # 使用字典推导式递归处理所有张量
        return {
            key: move_to_cuda(value) for key, value in data_dict.items()
        }
    else:
        print("CUDA is not available.")
        return data_dict

def move_to_cuda(value):
    """检查值并移动到 CUDA"""
    if isinstance(value, torch.Tensor):
        return value.cuda()
    elif isinstance(value, list):
        return [move_to_cuda(item) for item in value]
    elif isinstance(value, dict):
        return move_tensors_to_cuda(value)  # 递归调用以处理嵌套字典
    else:
        return value  # 其他类型保持不变

class Exec(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Store the input point clouds
        self.ref_path = self.args.ref_path
        self.src_path = self.args.src_path

        self.data_dict = {
            'seq_id': 0,
            'ref_frame': self.ref_path,
            'src_frame': self.src_path,
            'ref_points': self._load_point_cloud(self.ref_path),
            'src_points': self._load_point_cloud(self.src_path),
            'testing': True
        }
        self.data_dict['ref_feats'] = np.ones((self.data_dict['ref_points'].shape[0], 1), dtype=np.float32)
        self.data_dict['src_feats'] = np.ones((self.data_dict['src_points'].shape[0], 1), dtype=np.float32)

        cfg.neighbor_limits = calibrate_neighbors_stack_mode(
            [self.data_dict],
            registration_collate_fn_stack_mode,
            cfg.backbone.num_stages,
            cfg.backbone.init_voxel_size,
            cfg.backbone.init_radius,
        )
        self.data_dict = registration_collate_fn_stack_mode(
            [self.data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, cfg.neighbor_limits, precompute_data=True
        )
        self.data_dict = move_tensors_to_cuda(self.data_dict)
        # model
        self.model = create_model(cfg).cuda()
        self.model.training = False
        self.register_model(self.model)

        
    def _load_point_cloud(self, file_name):
        if file_name.endswith('.npy'):
            points = np.load(file_name)
        if file_name.endswith('.pcd'):
            pcd = o3d.io.read_point_cloud(file_name)
            points = np.array(pcd.points).astype(np.float32)
        # if self.point_limit is not None and points.shape[0] > self.point_limit:
        #     indices = np.random.permutation(points.shape[0])[: self.point_limit]
        #     points = points[indices]
        return points

    def test_step(self):
        # Create a data_dict for input
        
        
        
        output_dict = self.model(self.data_dict)
        return output_dict

    def after_test_step(self, output_dict):
        estimated_transform = release_cuda(output_dict['estimated_transform'])
        
        # Reshape the estimated_transform if needed
        print(estimated_transform)
        return estimated_transform

def main():
    cfg = make_cfg()
    

    # Create a Tester instance with the point clouds
    tester = Exec(cfg)
    
    # Perform the test step
    output_dict = tester.test_step()
    
    # Process the output to get estimated_transform
    estimated_transform = tester.after_test_step(output_dict)
    
    print("Estimated Transform:\n", estimated_transform)

if __name__ == '__main__':
    # e.g.:
    # python experiments/exec.py --src_path ../data/training_data/zhongtai/pairs_data/downsampled_xyzi/A1-016-2024-03-08-14-25-16_3189.npy 
    #                            --ref_path ../data/training_data/zhongtai/pairs_data/downsampled_xyzi/A1-016-2024-04-09-13-16-17_12081.npy
    main()
