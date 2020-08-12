# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li
# --------------------------------------------------------
import numpy as np
from .projection import se3_inverse, se3_mul, backproject_camera


def calc_flow(depth_src,
              pose_src,
              pose_tgt,
              K,
              depth_tgt,
              thresh=3e-3,
              standard_rep=False): 
    """
    project the points in source corrd to target corrd
    :param standard_rep:
    :param depth_src: depth image of source(m)
    :param pose_src: pose matrix of soucre, [R|T], 3x4
    :param depth_tgt: depth image of target
    :param pose_tgt: pose matrix of target, [R|T], 3x4
    :param K: intrinsic_matrix
    :param depth_tgt: depth image of target(m)
    :return: visible: whether points in source can be viewed in target
    :return: flow: flow from source to target
    """
    height = depth_src.shape[0]
    width = depth_src.shape[1]
    visible = np.zeros(depth_src.shape[:2]).flatten()
    X = backproject_camera(depth_src, intrinsic_matrix=K)
    transform = np.matmul(K, se3_mul(pose_tgt, se3_inverse(pose_src)))
    Xp = np.matmul(
        transform,
        np.append(X, np.ones([1, X.shape[1]], dtype=np.float32), axis=0))

    pz = Xp[2] + 1e-15
    pw = Xp[0] / pz
    ph = Xp[1] / pz

    valid_points = np.where(depth_src.flatten() != 0)[0]
    depth_proj_valid = pz[valid_points]
    pw_valid_raw = np.round(pw[valid_points]).astype(int)
    pw_valid = np.minimum(np.maximum(pw_valid_raw, 0), width - 1)
    ph_valid_raw = np.round(ph[valid_points]).astype(int)
    ph_valid = np.minimum(np.maximum(ph_valid_raw, 0), height - 1)
    p_within = np.logical_and(
        np.logical_and(pw_valid_raw >= 0, pw_valid_raw < width),
        np.logical_and(ph_valid_raw >= 0, ph_valid_raw < height),
    )

    depth_tgt_valid = depth_tgt[ph_valid, pw_valid]

    p_within = np.logical_and(
        p_within,
        np.abs(depth_tgt_valid - depth_proj_valid) < thresh)
    p_valid = np.abs(depth_tgt_valid) > 1e-10
    fg_points = valid_points[np.logical_and(p_within, p_valid)]
    visible[fg_points] = 1
    visible = visible.reshape(depth_src.shape[:2])
    w_ori, h_ori = np.meshgrid(np.linspace(0, width - 1, width),
                               np.linspace(0, height - 1, height))
    if standard_rep:
        flow = np.dstack([
            pw.reshape(depth_src.shape[:2]) - w_ori,
            ph.reshape(depth_src.shape[:2]) - h_ori
        ])
    else:
        # depleted version, only used in old code
        flow = np.dstack([
            ph.reshape(depth_src.shape[:2]) - h_ori,
            pw.reshape(depth_src.shape[:2]) - w_ori
        ])
    flow[np.dstack([visible, visible]) != 1] = 0
    assert np.isnan(flow).sum() == 0
    X_valid = np.array([c[np.where(visible.flatten())] for c in X])
    return flow, visible, X_valid