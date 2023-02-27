import os.path as osp
import numpy as np
import argparse
import json
import cv2
import trimesh
import open3d as o3d

try:
    import smplx
    import torch
except ImportError:
    print('smplx and torch are needed for visualization of SMPL vertices.')

from visualizer import transform_points


def visualize_3d(root_dir, seq_name, kinect_id, frame_id,
              visualize_smpl=False, smpl_model_path=None):
    """
    Args:
        root_dir (str): root directory in which data is stored.
        seq_name (str): sequence name, in the format 'pxxxxxx_axxxxxx'.
        kinect_id (int): Kinect ID. Available range is [0, 9].
        frame_id (int): frame ID. Available range varies for different sequences.
        visualize_smpl (bool): whether to visualize SMPL model. Defaults to False.
        smpl_model_path (str): directory in which SMPL body models are stored.
    Returns:
        None
    """
    assert frame_id % 6 == 0, 'Frame ID should be multiples of 6.'

    visual = []

    # load depth image
    depth_image_path = osp.join(root_dir, seq_name, 'kinect_depth', f'kinect_{kinect_id:03d}', f'{frame_id:06d}.png')
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    depth_image = o3d.geometry.Image(depth_image)

    # load depth camera parameters
    camera_path = osp.join(root_dir, seq_name, 'cameras.json')
    with open(camera_path, 'r') as f:
        cameras = json.load(f)
    camera_name = f'kinect_depth_{kinect_id:03d}'
    camera_params = cameras[camera_name]
    K, R, T = [np.array(camera_params[param]) for param in ['K', 'R', 'T']]

    open3d_camera = o3d.camera.PinholeCameraParameters()
    open3d_camera.intrinsic.set_intrinsics(width=640, height=576, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])

    open3d_point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
        depth_image, open3d_camera.intrinsic, depth_trunc=5.0)

    visual.append(open3d_point_cloud)

    # visualize SMPL model in 3D
    if visualize_smpl:

        # initialize SMPL body model
        smpl = smplx.create(
            model_path=smpl_model_path,
            model_type='smpl',
            gender='neutral')

        # load SMPL parameters in the world coordinate system
        smpl_params_path = osp.join(root_dir, seq_name, 'smpl_params', f'{frame_id:06d}.npz')
        smpl_params = np.load(smpl_params_path)
        global_orient = smpl_params['global_orient']
        body_pose = smpl_params['body_pose']
        betas = smpl_params['betas']
        transl = smpl_params['transl']

        # compute the SMPL vertices in the world coordinate system
        output = smpl(
            betas=torch.Tensor(betas).view(1, 10),
            body_pose=torch.Tensor(body_pose).view(1, 23, 3),
            global_orient=torch.Tensor(global_orient).view(1, 1, 3),
            transl=torch.Tensor(transl).view(1, 3),
            return_verts=True
        )
        vertices = output.vertices.detach().numpy().squeeze()

        # transform the vertices to the camera coordinate system
        vertices_cam = transform_points(vertices, R, T)

        # build 3D mesh
        faces = smpl.faces
        trimesh_mesh = trimesh.Trimesh(vertices_cam, faces, process=False)
        open3d_mesh = trimesh_mesh.as_open3d
        open3d_mesh.compute_vertex_normals()

        visual.append(open3d_mesh)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    visual.append(axis)
    o3d.visualization.draw_geometries(visual)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type=str,
                        help='root directory in which data is stored.')
    parser.add_argument('seq_name', type=str,
                        help='sequence name, in the format \'pxxxxxx_axxxxxx\'.')
    parser.add_argument('kinect_id', type=int, choices=list(range(10)),
                        help='Kinect ID. Available range is [0, 9]')
    parser.add_argument('frame_id', type=int,
                        help='available range varies for different sequences and should be multiples of 6.')
    parser.add_argument('--visualize_smpl', action='store_true',
                        help='whether to visualize SMPL model.')
    parser.add_argument('--smpl_model_path',
                        default='/home/user/mmhuman3d/data/body_models',
                        help="directory in which SMPL body models are stored")
    args = parser.parse_args()

    visualize_3d(
        root_dir=args.root_dir,
        seq_name=args.seq_name,
        kinect_id=args.kinect_id,
        frame_id=args.frame_id,
        visualize_smpl=args.visualize_smpl,
        smpl_model_path=args.smpl_model_path,
    )
