import numpy as np
import pybullet as pb
import transformations
from copy import deepcopy

"""
Transformation Utils
"""


def pose_to_mat(pose):
    orientation = deepcopy(pose['orientation'])
    if len(orientation) == 4:
        orientation = pb.getEulerFromQuaternion(orientation)
    return transformations.compose_matrix(angles=orientation, translate=pose['position'])


def mat_to_pose(mat, euler=False):
    _, _, orientation, position, _ = transformations.decompose_matrix(mat)
    if not euler:
        orientation = pb.getQuaternionFromEuler(orientation)
    return {'position': position, 'orientation': orientation}


def quaternion_avg_markley(Q, weights):
    '''
    https://stackoverflow.com/a/49690919
    Averaging Quaternions.
    Arguments:
        Q(ndarray): an Mx4 ndarray of quaternions.
        weights(list): an M elements list, a weight for each quaternion.
    '''

    # Form the symmetric accumulator matrix
    A = np.zeros((4, 4))
    M = Q.shape[0]
    wSum = 0

    for i in range(M):
        q = Q[i, :]
        w_i = weights[i]
        A += w_i * (np.outer(q, q))  # rank 1 update
        wSum += w_i

    # scale
    A /= wSum

    # Get the eigenvector corresponding to largest eigen value
    return np.linalg.eigh(A)[1][:, -1]