""" A simple visualization tool for HuMMan v1.0 """

import os.path as osp
import numpy as np
import argparse
import json
import cv2
try:
    import smplx
    import torch
except ImportError:
    print('smplx and torch are needed for visualization of SMPL vertices.')


def perspective_projection(points, K, R, T):
    """
    Args:
        points (np.ndarray): 3D points in world coordinate system of shape (N, 3).
        K (np.ndarray): camera intrinsic matrix of shape (3, 3).
        R (np.ndarray): world2cam rotation matrix of shape (3, 3).
        T (np.ndarray): world2cam translation vector of shape (3,).

    Returns:
        proj_points (np.ndarray): 2D points of shape (N, 2).
    """
    N = points.shape[0]

    # compute world to camera transformation
    T_world2cam = np.eye(4)
    T_world2cam[:3, :3] = R
    T_world2cam[:3, 3] = T

    # convert 3D points to homogeneous coordinates
    points_3D = points.T  # (3, N)
    points_homo = np.vstack([points_3D, np.ones((1, N))])  # (4, N)

    # transform points to the camera frame
    points_3D_cam = T_world2cam @ points_homo  # (4, N)
    points_3D_cam = points_3D_cam[:3, :]  # (3, N)

    # project to image plane
    points_2D = K @ points_3D_cam  # (3, N)
    points_2D = points_2D[:2, :] / points_2D[2, :]  # (2, N)
    proj_points = points_2D.T  # (N, 2)

    return proj_points


def visualize(root_dir, seq_name, kinect_id, frame_id,
              visualize_mask=False, visualize_smpl=False, smpl_model_path=None):
    """
    Args:
        root_dir (str): root directory in which data is stored.
        seq_name (str): sequence name, in the format 'pxxxxxx_axxxxxx'.
        kinect_id (int): Kinect ID. Available range is [0, 9].
        frame_id (int): frame ID. Available range varies for different sequences.
        visualize_mask (bool): whether to overlay mask on color image. Defaults to False.
        visualize_smpl (bool): whether to overlay SMPL vertices on color image. Defaults to False.
        smpl_model_path (str): directory in which SMPL body models are stored.
    Returns:
        None
    """
    # load color image
    color_path = osp.join(root_dir, seq_name, 'kinect_color', f'kinect_{kinect_id:03d}', f'{frame_id:06d}.png')
    color = cv2.imread(color_path)

    # overlay mask
    if visualize_mask:
        mask_path = osp.join(root_dir, seq_name, 'kinect_mask', f'kinect_{kinect_id:03d}', f'{frame_id:06d}.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        color[mask == 0] = 0

    # overlay SMPL vertices
    if visualize_smpl:

        # initialize SMPL body model
        smpl = smplx.create(
            model_path=smpl_model_path,
            model_type='smpl',
            gender='neutral')

        # load SMPL parameters in world coordinate system
        smpl_params_path = osp.join(root_dir, seq_name, 'smpl_params', f'{frame_id:06d}.npz')
        smpl_params = np.load(smpl_params_path)
        global_orient = smpl_params['global_orient']
        body_pose = smpl_params['body_pose']
        betas = smpl_params['betas']
        transl = smpl_params['transl']

        # computer SMPL vertices in world coordinate system
        output = smpl(
            betas=torch.Tensor(betas).view(1, 10),
            body_pose=torch.Tensor(body_pose).view(1, 23, 3),
            global_orient=torch.Tensor(global_orient).view(1, 1, 3),
            transl=torch.Tensor(transl).view(1, 3),
            return_verts=True
        )
        vertices = output.vertices.detach().numpy().squeeze()

        # load camera parameters
        camera_path = osp.join(root_dir, seq_name, 'cameras.json')
        with open(camera_path, 'r') as f:
            cameras = json.load(f)
        camera_name = f'kinect_color_{kinect_id:03d}'
        camera_params = cameras[camera_name]
        K, R, T = camera_params['K'], camera_params['R'], camera_params['T']

        # project vertices
        proj_vertices = perspective_projection(vertices, K, R, T)

        # draw on the color image
        proj_vertices = np.round(proj_vertices).astype(int)
        for point in proj_vertices:
            color = cv2.circle(color, point, 1, (255, 255, 255), thickness=-1)

    cv2.namedWindow('Visualizer', cv2.WINDOW_NORMAL)
    cv2.imshow('Visualizer', color)
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type=str,
                        help='root directory in which data is stored.')
    parser.add_argument('seq_name', type=str,
                        help='sequence name, in the format \'pxxxxxx_axxxxxx\'.')
    parser.add_argument('kinect_id', type=int, choices=list(range(10)),
                        help='Kinect ID. Available range is [0, 9]')
    parser.add_argument('frame_id', type=int,
                        help='Available range varies for different sequences and should be multiples of 6.')
    parser.add_argument('--visualize_mask', action='store_true',
                        help='whether to overlay mask on color image.')
    parser.add_argument('--visualize_smpl', action='store_true',
                        help='whether to overlay SMPL vertices on color image.')
    parser.add_argument('--smpl_model_path',
                        default='/home/user/mmhuman3d/data/body_models',
                        help="directory in which SMPL body models are stored")
    args = parser.parse_args()

    visualize(
        root_dir=args.root_dir,
        seq_name=args.seq_name,
        kinect_id=args.kinect_id,
        frame_id=args.frame_id,
        visualize_mask=args.visualize_mask,
        visualize_smpl=args.visualize_smpl,
        smpl_model_path=args.smpl_model_path,
    )
