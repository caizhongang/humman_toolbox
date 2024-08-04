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


def perspective_projection(points, K):
    """ Project 3D points in camera coordinate system onto the camera plane
    Args:
        points (np.ndarray): 3D points in camera coordinate system of shape (N, 3).
        K (np.ndarray): camera intrinsic matrix of shape (3, 3).

    Returns:
        proj_points (np.ndarray): 2D points of shape (N, 2).
    """
    points = points.T  # (3, N)

    # project to image plane
    points_2D = K @ points  # (3, N)
    points_2D = points_2D[:2, :] / points_2D[2, :]  # (2, N)
    proj_points = points_2D.T  # (N, 2)

    return proj_points


def transform_points(points, R, T):
    """ Transform 3D points from world coordinate system to camera coordinate system
    Args:
        points (np.ndarray): 3D points in world coordinate system of shape (N, 3).
        R (np.ndarray): world2cam rotation matrix of shape (3, 3).
        T (np.ndarray): world2cam translation vector of shape (3,).

    Returns:
        transformed_points (np.ndarray): 3D points in camera coordiante system of shape (N, 2).
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
    transformed_points = T_world2cam @ points_homo  # (4, N)
    transformed_points = transformed_points[:3, :]  # (3, N)
    transformed_points = transformed_points.T  # (N, 3)

    return transformed_points


def compute_color2depth_transform(color_camera_params, depth_camera_params):
    
    # Compute color camera transformation
    T_world2color = np.eye(4)
    T_world2color[:3, :3] = color_camera_params['R']
    T_world2color[:3, 3] = color_camera_params['T']

    # Compute depth camera transformation
    T_world2depth = np.eye(4)
    T_world2depth[:3, :3] = depth_camera_params['R']
    T_world2depth[:3, 3] = depth_camera_params['T']
     
    # Compute depth2color transformation
    T_color2depth = T_world2depth @ np.linalg.inv(T_world2color)

    return T_color2depth


def create_rgbd(color_image, depth_image, color_camera_params, depth_camera_params):
    """ Create RGB-D images with color/depth images and their parameters.
        Note for Kinects, their color/depth cameras are not perfectly aligned. 
    Args:
        color_image (np.ndarray): color image (RGB)
        depth_iamge (np.ndarray): depth image
        color_camera_params (dict): camera parameters for the color image
        depth_camera_params (dict): camera parameters for the depth image
    Returns:
        rgbd (): colored point clouds
    """
    # Create point clouds from the depth image
    open3d_camera = o3d.camera.PinholeCameraParameters()
    open3d_camera.intrinsic.set_intrinsics(
        # width=640, height=576, 
        width=1920, height=1440, 
        fx=depth_camera_params['K'][0, 0] / 7.5, 
        fy=depth_camera_params['K'][1, 1] / 7.5, 
        cx=depth_camera_params['K'][0, 2] / 7.5, 
        cy=depth_camera_params['K'][1, 2] / 7.5)
    import pdb; pdb.set_trace()
    # depth_image = cv2.resize(depth_image, (1920, 1440), interpolation=cv2.INTER_LINEAR)
    depth_image = o3d.geometry.Image(depth_image)
    open3d_point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
        depth_image, open3d_camera.intrinsic, depth_trunc=5.0)

    o3d.visualization.draw_geometries([open3d_point_cloud])

    T_color2depth = compute_color2depth_transform(color_camera_params, depth_camera_params)
    open3d_point_cloud.transform(T_color2depth)

    # Project the point cloud to the color image
    projected_points = perspective_projection(np.array(open3d_point_cloud.points), color_camera_params['K'])

    # Get the colors from the RGB image
    colors = []
    for pt in projected_points:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < color_image.shape[1] and 0 <= y < color_image.shape[0]:
            colors.append(color_image[y, x] / 255.0)
        else:
            colors.append([0, 0, 0])  # Default color for out of bounds

    # Assign the colors to the point cloud
    open3d_point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))

    return open3d_point_cloud


# def visulize_rgbd_video(root_dir, seq_name, device_name, visualize_smpl=False, smpl_model_path=None):
    
#     # load color video
#     color_video_path = osp.join(root_dir, seq_name, 'color', '.mp4')
#     cap = cv2.VideoCapture(color_video_path)
#     color_images = []
#     while cap.isOpened():
#         success, color = cap.read()
#         if not success:
#             break
#         color_images.append(color)
#     cap.release()

#     # load depth image
#     depth_image_paths = sorted(glob.glob(osp.join(root_dir, seq_name, 'depth', device_name, '*.png'))
#     depth_images = []
#     for depth_image_path in depth_image_paths:
#         depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

#     for color_image, depth_image in zip(color_images, depth_images):


def visualize_rgbd_interactive(root_dir, seq_name, device_name, frame_id=0, visualize_smpl=False, smpl_model_path=None):
    """ Visualize oen frame of RGB-D data with an interactive viewer (adjustable camera pose).
    Args:
        root_dir (str): root directory in which data is stored.
        seq_name (str): sequence name, in the format 'pxxxxxx_axxxxxx'.
        device_name (str): device name. 'kinect_000' to 'kinect_009', and 'iphone' are available.
        frame_id (int): frame ID. Available range varies for different sequences. Defaults to 0.
        visualize_smpl (bool): whether to visualize SMPL 3D mesh model. Defaults to False.
        smpl_model_path (str): directory in which SMPL body models are stored.
    Returns:
        None
    """
    visual = []

    # load camera parameters
    camera_path = osp.join(root_dir, seq_name, 'cameras.json')
    with open(camera_path, 'r') as f:
        cameras = json.load(f)
    if 'kinect' in device_name:
        color_camera_name = f'kinect_color_{device_name[-3:]}'
        depth_camera_name = f'kinect_depth_{device_name[-3:]}'
        color_camera_params = {k: np.array(v) for k, v in cameras[color_camera_name].items()}
        depth_camera_params = {k: np.array(v) for k, v in cameras[depth_camera_name].items()}
    else:
        color_camera_params = depth_camera_params = {k: np.array(v) for k, v in cameras['iphone'].items()}

    # load color image
    device = 'kinect' if 'kinect' in device_name else 'iphone'
    color_video_path = osp.join(root_dir, seq_name, f'{device}_color', f'{device_name}.mp4')
    cap = cv2.VideoCapture(color_video_path)
    assert cap.isOpened(), f'Error opening video file: {color_video_path}'
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    success, color_image = cap.read()
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    assert success, f'Error reading frame {frame_id} from {color_video_path}'
    cap.release()
    
    # load depth image
    depth_image_path = osp.join(root_dir, seq_name, f'{device}_depth', device_name, f'{frame_id:06d}.png')
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    import pdb; pdb.set_trace()
    
    # Genearte RGB-D point cloud
    rgbd_point_cloud = create_rgbd(color_image, depth_image, color_camera_params, depth_camera_params)
    visual.append(rgbd_point_cloud)

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
        vertices_cam = transform_points(
            vertices, color_camera_params['R'], color_camera_params['T'])

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
    parser.add_argument('device_name', type=str, 
                        choices=['kinect_000', 'kinect_001', 'kinect_002', 'kinect_003',
                                 'kinect_004', 'kinect_005', 'kinect_006', 'kinect_007',
                                 'kinect_008', 'kinect_009', 'iphone'],
                        help='device name. \'kinect_000\' to \'kinect_009\', and \'iphone\' are available.')
    parser.add_argument('--frame_id', type=int, default=-1,
                        help='frame id. If not specified, the entire video will be visualized.')
    parser.add_argument('--virtual_cam', type=str, default='assets/virtual_cam.json',
                        help='virtual camera pose. Required for visualizing the entire video.')
    parser.add_argument('--visualize_smpl', action='store_true',
                        help='whether to visualize SMPL 3D mesh model.')
    parser.add_argument('--smpl_model_path',
                        default='/home/user/mmhuman3d/data/body_models',
                        help="directory in which SMPL body models are stored.")
    args = parser.parse_args()

    if args.frame_id == -1:
        raise NotImplementedError('Visualizing the entire video is not supported yet.')
        # visualize_rgbd_video(
        # )
    else:    
        visualize_rgbd_interactive(
            root_dir=args.root_dir,
            seq_name=args.seq_name,
            device_name=args.device_name,
            frame_id=args.frame_id,
            visualize_smpl=args.visualize_smpl,
            smpl_model_path=args.smpl_model_path,
        )

        