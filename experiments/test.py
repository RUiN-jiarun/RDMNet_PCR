import argparse
import os.path as osp
import time

import numpy as np

from geotransformer.engine import SingleTester
from geotransformer.utils.common import ensure_dir, get_log_string
from geotransformer.utils.torch import release_cuda
from geotransformer.utils.registration import (
    evaluate_sparse_correspondences,
    evaluate_correspondences,
    compute_registration_error,
    get_rotation_translation_from_transform,
    evaluate_overlap,
    evaluate_node_overlap
)
from geotransformer.utils.common import rotmat2qvec

# from dataset import test_data_loader
from dataset import test_data_loader

from loss import Evaluator
from model import create_model
from config import make_cfg


class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg)

        # dataloader
        start_time = time.time()
        data_loader, neighbor_limits = test_data_loader(cfg, self.distributed, cfg.dataset)
        cfg.neighbor_limits=neighbor_limits
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        self.logger.info(message)
        message = f'Calibrate neighbors: {neighbor_limits}.'
        self.logger.info(message)
        self.register_loader(data_loader)

        # model
        model = create_model(cfg).cuda()
        self.register_model(model)

        # evaluator
        self.evaluator = Evaluator(cfg).cuda()

        # preparation
        self.output_dir = osp.join(cfg.feature_dir)
        ensure_dir(self.output_dir)

    def test_step(self, iteration, data_dict):
        data_dict['testing'] = True
        output_dict = self.model(data_dict)
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        data_dict['testing'] = True
        data_dict['evaling'] = True
        result_dict = self.evaluator(output_dict, data_dict)
        return result_dict

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        seq_id = data_dict['seq_id']
        ref_frame = data_dict['ref_frame']
        src_frame = data_dict['src_frame']
        message = f'seq_id: {seq_id}, id0: {ref_frame}, id1: {src_frame}'
        message += ', ' + get_log_string(result_dict=result_dict)
        message += ', nCorr: {}'.format(output_dict['corr_scores'].shape[0])
        return message

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        seq_id = data_dict['seq_id']
        ref_frame = data_dict['ref_frame']
        src_frame = data_dict['src_frame']

        gt_transform = release_cuda(data_dict['transform'])
        estimated_transform = release_cuda(output_dict['estimated_transform'])
        est_rotation, est_translation = get_rotation_translation_from_transform(estimated_transform)
        gt_rotation, gt_translation = get_rotation_translation_from_transform(gt_transform)
        gt_quat = rotmat2qvec(gt_rotation)
        est_quat = rotmat2qvec(est_rotation)
        rre, rte, rx, ry, rz = compute_registration_error(gt_transform, estimated_transform)
        accepted = (rre < 5) and (rte < 0.1)

        message = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(iteration, src_frame, ref_frame, est_quat[0], est_quat[1], est_quat[2], est_quat[3], est_translation[0], est_translation[1], est_translation[2], 
                                                            gt_quat[0], gt_quat[1], gt_quat[2], gt_quat[3], gt_translation[0], gt_translation[1], gt_translation[2], rre, rte, accepted)
        self.logger.info(message)

        # file_name = osp.join(self.output_dir, f'{seq_id}_{src_frame}_{ref_frame}.npz')
        # np.savez_compressed(
        #     file_name,
        #     ref_points=release_cuda(output_dict['ref_points']),
        #     src_points=release_cuda(output_dict['src_points']),
        #     ref_points_f=release_cuda(output_dict['ref_points_f']),
        #     src_points_f=release_cuda(output_dict['src_points_f']),
        #     ref_points_c=release_cuda(output_dict['ref_points_c']),
        #     src_points_c=release_cuda(output_dict['src_points_c']),
        #     ref_feats_c=release_cuda(output_dict['ref_feats_c']),
        #     src_feats_c=release_cuda(output_dict['src_feats_c']),
        #     ref_node_corr_indices=release_cuda(output_dict['ref_node_corr_indices']),
        #     src_node_corr_indices=release_cuda(output_dict['src_node_corr_indices']),
        #     ref_corr_points=release_cuda(output_dict['ref_corr_points']),
        #     src_corr_points=release_cuda(output_dict['src_corr_points']),
        #     corr_scores=release_cuda(output_dict['corr_scores']),
        #     gt_node_corr_indices=release_cuda(output_dict['gt_node_corr_indices']),
        #     gt_node_corr_overlaps=release_cuda(output_dict['gt_node_corr_overlaps']),
        #     estimated_transform=release_cuda(output_dict['estimated_transform']),
        #     transform=release_cuda(data_dict['transform']),
            
        #     # for visualization
        #     # ori_ref_points_c=release_cuda(output_dict['ori_ref_points_c']),
        #     # ori_src_points_c=release_cuda(output_dict['ori_src_points_c']),
        #     # shifted_ref_points_c=release_cuda(output_dict['shifted_ref_points_c']),
        #     # shifted_src_points_c=release_cuda(output_dict['shifted_src_points_c']),
        #     # ref_frame=release_cuda(data_dict['ref_frame']),
        #     # src_frame=release_cuda(data_dict['src_frame']),
        # )


def main():

    cfg = make_cfg()
    
    cfg.feature_dir = cfg.feature_dir + f'{cfg.dataset}'

    if cfg.dataset=='mulran':
        cfg.Vote.inference_use_vote = False

    tester = Tester(cfg)
    tester.run()


if __name__ == '__main__':
    main()