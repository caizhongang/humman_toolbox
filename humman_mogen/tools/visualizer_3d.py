import open3d as o3d
import smplx
import trimesh
import torch
import json
import cv2
import numpy as np
import os.path as osp
import os
import argparse

camera_load_path = './tools/camera.json'
angle_per_frame = 1  # degree
fps = 30

def visualize(smpl_load_path, video_save_dir, smpl_model_path, draw_ground, draw_axis):
    """ Visualize SMPL on ground plane """
    stem, _ = osp.splitext(osp.basename(smpl_load_path))
    os.makedirs(video_save_dir, exist_ok=True)
    video_save_path = osp.join(video_save_dir, stem + '.mp4')

    # if osp.exists(video_save_path):
    #     return

    with open(camera_load_path, 'r') as f:
        camera_params = json.load(f)
    height, width = camera_params['intrinsic']['height'], camera_params['intrinsic']['width']

    smpl = smplx.create(
        model_path=smpl_model_path,
        model_type='smpl',
        gender='neutral')

    smpl_params = np.load(smpl_load_path, allow_pickle=True)
    num_frames = len(smpl_params['transl'])
    
    transl = smpl_params['transl']
    body_pose = smpl_params['body_pose']
    global_orient = smpl_params['global_orient']
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=height, width=width)
    ctr = vis.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters(camera_load_path)

    frames = []
    for i in range(num_frames):
        frame_global_orient = global_orient[i]
        frame_body_pose = body_pose[i]
        frame_betas = smpl_params['betas'][i]
        frame_transl = transl[i]
        output = smpl(
            betas=torch.Tensor(frame_betas).view(1, 10),
            body_pose=torch.Tensor(frame_body_pose).view(1, 23, 3),
            global_orient=torch.Tensor(frame_global_orient).view(1, 1, 3),
            transl=torch.Tensor(frame_transl).view(1, 3),
            return_verts=True
        )
        vertices = output.vertices.detach().numpy().squeeze()
        pelvis = output.joints[0, 0].detach().numpy().squeeze()
        faces = smpl.faces
        trimesh_mesh = trimesh.Trimesh(vertices, faces, process=False)
        open3d_mesh = trimesh_mesh.as_open3d
        open3d_mesh.compute_vertex_normals()

        obj_list = [open3d_mesh]
        if draw_ground:
            ground = o3d.geometry.TriangleMesh.create_box(width=30.0, height=30.0, depth=0.01).translate([-15, -15, -0.01])
            obj_list.append(ground)
        if draw_axis:
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
            obj_list.append(axis)
        
        for obj in obj_list:
            R = obj.get_rotation_matrix_from_xyz([0, 0, i * np.deg2rad(angle_per_frame)])
            obj.rotate(R, center=pelvis) 

        for obj in obj_list:
            vis.add_geometry(obj)

        ctr.convert_from_pinhole_camera_parameters(parameters) 

        for obj in obj_list:
            vis.update_geometry(obj)

        vis.poll_events()
        vis.update_renderer()

        frame = np.array(vis.capture_screen_float_buffer())
        frames.append(frame)

        for obj in obj_list:
            vis.remove_geometry(obj)
    
    out = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame in frames:
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type=str,
                        help='root directory in which smpl sequences (.npz) are stored.')
    parser.add_argument('seq_name', type=str,
                        help='sequence name, in the format \'pxxxxxx_axxxxxx\' or \'pxxxxxx_rxxxxxx\'. Input \'*\' to visualize all sequences under the root directory. ')
    parser.add_argument('video_dir', type=str,
                        help='directory to save rendered videos.')
    parser.add_argument('--draw_ground', action='store_true',
                        help='whether to visualize a ground plane.')
    parser.add_argument('--draw_axis', action='store_true',
                        help='whether to visualize the xyz axis.')
    parser.add_argument('--smpl_model_path',
                        default='/home/user/body_models',
                        help="directory in which SMPL body models are stored.")
    args = parser.parse_args()
 
    root_dir = args.root_dir

    fout = open("log.txt", "w")
    if args.seq_name != '*':
        try:
            visualize(osp.join(root_dir, args.seq_name + '.npz'), args.video_dir, args.smpl_model_path, args.draw_ground, args.draw_axis)
        except:
            fout.write(args.seq_name + " Failed\n")
            fout.flush()
    else:
        for filename in os.listdir(root_dir):
            try:
                visualize(osp.join(root_dir, filename), args.video_dir, args.smpl_model_path, args.draw_ground, args.draw_axis)
            except:
                fout.write(filename.split('.')[0] + " Failed\n")
                fout.flush()
        