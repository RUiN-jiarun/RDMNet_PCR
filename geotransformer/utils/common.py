import os
import os.path as osp
import pickle
import numpy as np

def ensure_dir(path):
    if not osp.exists(path):
        os.makedirs(path)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def dump_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def get_print_format(value):
    if isinstance(value, int):
        return 'd'
    if isinstance(value, str):
        return 's'
    if value == 0:
        return '.3f'
    if value < 1e-6:
        return '.3e'
    if value < 1e-3:
        return '.6f'
    return '.3f'


def get_format_strings(kv_pairs):
    r"""Get format string for a list of key-value pairs."""
    log_strings = []
    for key, value in kv_pairs:
        fmt = get_print_format(value)
        format_string = '{}: {:' + fmt + '}'
        log_strings.append(format_string.format(key, value))
    return log_strings


def get_log_string(result_dict, epoch=None, max_epoch=None, iteration=None, max_iteration=None, lr=None, timer=None):
    log_strings = []
    if epoch is not None:
        epoch_string = f'Epoch: {epoch}'
        if max_epoch is not None:
            epoch_string += f'/{max_epoch}'
        log_strings.append(epoch_string)
    if iteration is not None:
        iter_string = f'iter: {iteration}'
        if max_iteration is not None:
            iter_string += f'/{max_iteration}'
        if epoch is None:
            iter_string = iter_string.capitalize()
        log_strings.append(iter_string)
    if 'metadata' in result_dict:
        log_strings += result_dict['metadata']
    for key, value in result_dict.items():
        if key != 'metadata':
            format_string = '{}: {:' + get_print_format(value) + '}'
            log_strings.append(format_string.format(key, value))
    if lr is not None:
        log_strings.append('lr: {:.3e}'.format(lr))
    if timer is not None:
        log_strings.append(timer.tostring())
    message = ', '.join(log_strings)
    return message

def qvec2rotmat(qvec):
    return np.array([[
        1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
        2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
        2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
    ],
    [
        2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
        1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
        2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
    ],
    [
        2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
        2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
        1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
    ]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([[Rxx - Ryy - Rzz, 0, 0, 0], [
        Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0
    ], [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                  [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def generte_to_matrix(qvec, tvec):
    trans_mat = np.identity(4)
    trans_mat[0:3, 0:3] = qvec2rotmat(qvec)
    trans_mat[0:3, 3] = tvec
    return trans_mat

def load_metadata(data_root, subset):
    id_enum = {
        'zhongtai': 0,
        'pingyao': 1,
        'nanhu': 2,
    }
    pairs_pose_file = osp.join(data_root, subset, 'pairs_data', 'pair_pose.txt')
    metadata = []
    with open(pairs_pose_file, 'r') as f:
        for line in f:
            data = line.strip().split()
            overlap = float(data[2])
            if overlap > 0.3 or overlap < 0.08:
                continue
            item = {}
            item['seq_id'] = id_enum[subset]
            item['overlap'] = overlap
            item['frame0'] = data[5] + '_' + data[6]
            item['frame1'] = data[7] + '_' + data[8]
            item['pcd0'] = osp.join(subset, 'pairs_data', 'downsampled_xyzi', data[5] + '_' + data[6] + '.npy')
            item['pcd1'] = osp.join(subset, 'pairs_data', 'downsampled_xyzi', data[7] + '_' + data[8] + '.npy')
            # item['pcd0'] = osp.join(subset, 'pairs_data', 'submap', data[5] + '_' + data[6] + '.pcd')
            # item['pcd1'] = osp.join(subset, 'pairs_data', 'submap', data[7] + '_' + data[8] + '.pcd')
            qvec = np.array(data[9:13], dtype=np.float32)
            qvec = qvec[[3, 0, 1, 2]]
            tvec = np.array(data[13:], dtype=np.float32)
            item['transform'] = generte_to_matrix(qvec, tvec)
            metadata.append(item)
    return metadata