# Toolbox for HuMMan Dataset

Please visit our [Homepage](https://caizhongang.github.io/projects/HuMMan/) for more details.      

## Updates
- [2023-02-27] Release of depth maps for the Reconstruction Subset
- [2023-02-27] Downloads are organized by modalities, links have been updated
- [2023-01-23] Release of textured meshes for the Reconstruction Subset, and a toolbox
- [2023-01-23] Minor fixes on the mask data, download links have been updated
- [2023-01-11] Release of HuMMan v1.0: Reconstruction Subset
- [2022-10-27] We presented HuMMan as an oral paper at ECCV'22 (Tel Aviv, Israel)
- [2022-08] Release of HuMMan v0.1 (no longer available, please use v1.0)

## HuMMan Release v1.0: Reconstruction Subset

HuMMan v1.0: Reconstruction Subset consists of 153 subjects and 339 sequences. 
Color images, masks (via matting), SMPL parameters, and camera parameters are 
provided. It is a challenging dataset for its collection of diverse subject 
appearance and expressive actions. Moreover, it unleashes the potential to 
benchmark reconstruction algorithms under realistic settings with commercial 
sensors, dynamic subjects, and computer vision-powered automatic annotations.

### Downloads

Color images:
- Part 1: [Aliyun](https://openxdlab.oss-cn-shanghai.aliyuncs.com/HuMMan/humman_release_v1.0_recon/recon_kinect_color_part_1.zip) 
or [OneDrive(CN)](https://pjlab-my.sharepoint.cn/:u:/g/personal/openmmlab_pjlab_org_cn/Ed6CM2tKVmRGiqcCyIcd4rABYy8kcf_tIbDU4nmuM4Zi1Q?e=yrC0pT) 
(~84 GB)
- Part 2: [Aliyun](https://openxdlab.oss-cn-shanghai.aliyuncs.com/HuMMan/humman_release_v1.0_recon/recon_kinect_color_part_2.zip) 
or [OneDrive(CN)](https://pjlab-my.sharepoint.cn/:u:/g/personal/openmmlab_pjlab_org_cn/EY1aLq6U9ppJsYuMy_y30yABGhrepLtgk4KQ_9Xl_O3heQ?e=7cTTvM) 
(~73 GB)
- Part 3: [Aliyun](https://openxdlab.oss-cn-shanghai.aliyuncs.com/HuMMan/humman_release_v1.0_recon/recon_kinect_color_part_3.zip) 
or [OneDrive(CN)](https://pjlab-my.sharepoint.cn/:u:/g/personal/openmmlab_pjlab_org_cn/Edd6RwJ-9VJCpV99BKXp0SgBNvvtLdZmJEOvrFv_K8Aj1w?e=XkjLTu) 
(~84 GB)

Masks for color images: [Aliyun](https://openxdlab.oss-cn-shanghai.aliyuncs.com/HuMMan/humman_release_v1.0_recon/recon_kinect_mask.zip) 
or [OneDrive(CN)](https://pjlab-my.sharepoint.cn/:u:/g/personal/openmmlab_pjlab_org_cn/ERVZ8yDijvZItS7ptjT3JiIB4dE0QJ-8802bWdWGidBl1g?e=DnzSm1) 
(~2.1 GB)

Depth images: [Aliyun](https://openxdlab.oss-cn-shanghai.aliyuncs.com/HuMMan/humman_release_v1.0_recon/recon_kinect_depth.zip) 
or [OneDrive(CN)](https://pjlab-my.sharepoint.cn/:u:/g/personal/openmmlab_pjlab_org_cn/EU6npJQmLMxEv7qY1ps4TMcBdBtblBOfW2Kese8_xFSvkw?e=F5a2NX) 
(~20 GB)

SMPL parameters: [Aliyun](https://openxdlab.oss-cn-shanghai.aliyuncs.com/HuMMan/humman_release_v1.0_recon/recon_smpl_params.zip) 
or [OneDrive(CN)](https://pjlab-my.sharepoint.cn/:u:/g/personal/openmmlab_pjlab_org_cn/EbYzP89JCppHnGWn6WGyZUcBt9ZSRIZdJIm--cJVmu7XKw?e=Bq4HPp)
(~7.6 MB)

Camera parameters (world2cam): [Aliyun](https://openxdlab.oss-cn-shanghai.aliyuncs.com/HuMMan/humman_release_v1.0_recon/recon_cameras.zip) 
or [OneDrive(CN)](https://pjlab-my.sharepoint.cn/:u:/g/personal/openmmlab_pjlab_org_cn/EZyvqo2fKhdBmyLj6Tchf8MB_dUr2cneVwCdD29l6nyP2A?e=GsLBqF)
(~1.3 MB). 

Textured meshes: [Aliyun](https://openxdlab.oss-cn-shanghai.aliyuncs.com/HuMMan/humman_release_v1.0_recon/recon_textured_meshes.zip) 
or [OneDrive(CN)](https://pjlab-my.sharepoint.cn/:u:/g/personal/openmmlab_pjlab_org_cn/EYgbLNilus1Auut9N-qs5rkBZ-NuNYWG5ml-tRxTlgeohA?e=Sdaogm)
(~22 GB)

### Data Structure
Please download the .zip files and decompress them into the following file structure:
```text
humman_release_v1.0_recon/   
└── pxxxxxx_axxxxxx/  
    ├── kinect_color/
    │   ├── kinect_000/
    │   ...
    │   └── kinect_009/
    │       ├── 000000.png (uint8, 3 channels)
    │       ├── 000006.png
    │       ...
    │       └── xxxxxx.png
    │
    ├── kinect_mask/
    │   ├── kinect_000/
    │   ...
    │   └── kinect_009/
    │       ├── 000000.png (uint8, 1 channel)
    │       ├── 000006.png
    │       ...
    │       └── xxxxxx.png
    │
    ├── kinect_depth/
    │   ├── kinect_000/
    │   ...
    │   └── kinect_009/
    │       ├── 000000.png (uint16, 1 channel)
    │       ├── 000006.png
    │       ...
    │       └── xxxxxx.png
    │
    ├── smpl_params/
    │   ├── 000000.npz
    │   ├── 000006.npz
    │   ...
    │   └── xxxxxx.npz
    │
    ├── cameras.json
    │
    └── textured_meshes/
        ├── 000000.mtl
        ├── 000000.obj
        ├── 000000_0.png*
        ├── 000006.mtl
        ├── 000006.obj
        ├── 000006_0.png*
        ...
        ├── xxxxxx.mtl
        ├── xxxxxx.obj
        └── xxxxxx_0.png*
    
```
- \* indicates that there be multiple .png for one .obj file.

#### kinect_color/
```python
import cv2
color_bgr = cv2.imread('/path/to/xxxxxxx.png')
color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)  # if RGB images are used
```

#### kinect_mask/
```python
import cv2
mask = cv2.imread('/path/to/xxxxxxx.png', cv2.IMREAD_GRAYSCALE)  # grayscale
```

#### smpl_parms/
SMPL parameters are in the world coordinate system.
The world coordinate system is the same as kinect_color_000 coordinate system. 
Each .npz consists of the following:
```text
{
    'betas': np.array of shape (10,),
    'body_pose': np.array of shape (69,),
    'global_orient': np.array of shape (3,),
    'transl': np.array of shape (3,)
}
```

To read:
```python
import numpy as np
smpl_params = np.load('/path/to/xxxxxx.npz')
global_orient = smpl_params['global_orient']
body_pose = smpl_params['body_pose']
betas = smpl_params['betas']
transl = smpl_params['transl']
```

#### cameras.json
Cameras use the OpenCV pinhole camera model.
- K: intrinsic matrix
- R: rotation matrix
- T: translation vector

R, T form world2cam transformation.
The world coordinate system is the same as kinect_color_000 coordinate system. 

Each .json consists of the following:
```text
{
    "kinect_color_000": {
        "K": np.array of shape (3,3),
        "R": np.array of shape (3,3),
        "T": np.array of shape (3,)
    },
    ...
    "kinect_color_009": {...}
}
```

To read:
```python
import json
with open('/path/to/cameras.json', 'r') as f:
    cameras = json.load(f)
camera_params = cameras['kinect_color_000']  # e.g., Kinect ID = 0
K, R, T = camera_params['K'], camera_params['R'], camera_params['T']
```


#### kinect_depth/
To read the depth maps
```python
import cv2
depth_image = cv2.imread('/path/to/xxxxxxx.png', cv2.IMREAD_UNCHANGED)
```

To convert the depth maps to point clouds in camera coordinate system
```python
import open3d as o3d
import numpy as np
import json
import cv2

# load depth image
depth_image = cv2.imread('/path/to/xxxxxxx.png', cv2.IMREAD_UNCHANGED)

# load depth camera parameters
with open('/path/to/cameras.json', 'r') as f:
    cameras = json.load(f)
camera_params = cameras['kinect_depth_000']  # e.g., Kinect ID = 0
K, R, T = camera_params['K'], camera_params['R'], camera_params['T']

# initialize open3d camera
open3d_camera = o3d.camera.PinholeCameraParameters()
open3d_camera.intrinsic.set_intrinsics(
    width=640, height=576, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])

# generate point cloud
point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
    depth_image, open3d_camera.intrinsic)
point_cloud = np.array(point_cloud)
```

### Visualization
We provide a simple 2D visualization tool for color images, masks, and SMPL vertices,
and a simple 3D visualization tool for point clouds (from depth images) and SMPL mesh models.

#### Installation
The tool does not require specific version of dependency packages. 
The following is for reference only. 
```bash
conda create -n humman python=3.8 -y
conda activate humman
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge -y
pip install opencv-python smplx chumpy trimesh
pip install open3d  # additional package for 3D visualization
```

#### Run 2D Visualizer
```bash
python tools/visualizer <root_dir> <seq_name> <kinect_id> <frame_id> \
  [--visualize_mask] [--visualize_smpl] [--smpl_model_path]
```
- root_dir (str): root directory in which data is stored.
- seq_name (str): sequence name, in the format 'pxxxxxx_axxxxxx'.
- kinect_id (int): Kinect ID. Available range is [0, 9].
- frame_id (int): frame ID. Available range varies for different sequences.
- visualize_mask (bool, optional): whether to overlay mask on color image. Defaults to False.
- visualize_smpl (bool, optional): whether to overlay SMPL vertices on color image. Defaults to False.
- smpl_model_path (str, optional): directory in which SMPL body models are stored.

Example:
```bash
python tools/visualizer /home/user/humman_release_v1.0_recon p000455_a000986 0 0 \
  --visualize_mask --visualize_smpl --smpl_model_path /home/user/body_models/
```

Note that the SMPL model path should consist the following structure:
```text
body_models/   
└── smpl/  
    └── SMPL_NEUTRAL.pkl
```
`SMPL_NEUTRAL.pkl` may be downloaded from [here](https://github.com/classner/up/blob/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl). 
Renaming from `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` to `SMPL_NEUTRAL.pkl` is needed.


#### Run 3D Visualizer
```bash
python tools/visualizer_3d <root_dir> <seq_name> <kinect_id> <frame_id> \
  [--visualize_smpl] [--smpl_model_path]
```
- root_dir (str): root directory in which data is stored.
- seq_name (str): sequence name, in the format 'pxxxxxx_axxxxxx'.
- kinect_id (int): Kinect ID. Available range is [0, 9].
- frame_id (int): frame ID. Available range varies for different sequences.
- visualize_smpl (bool, optional): whether to overlay SMPL vertices on color image. Defaults to False.
- smpl_model_path (str, optional): directory in which SMPL body models are stored.

Example:
```bash
python tools/visualizer /home/user/humman_release_v1.0_recon p000455_a000986 0 0 \
  --visualize_smpl --smpl_model_path /home/user/body_models/
```


## Citation
Please cite our work if you find our dataset and tools useful.
```text
@inproceedings{cai2022humman,
  title={{HuMMan}: Multi-modal 4d human dataset for versatile sensing and modeling},
  author={Cai, Zhongang and Ren, Daxuan and Zeng, Ailing and Lin, Zhengyu and Yu, Tao and Wang, Wenjia and Fan, 
          Xiangyu and Gao, Yang and Yu, Yifan and Pan, Liang and Hong, Fangzhou and Zhang, Mingyuan and
          Loy, Chen Change and Yang, Lei and Liu, Ziwei},
  booktitle={17th European Conference on Computer Vision, Tel Aviv, Israel, October 23--27, 2022, 
             Proceedings, Part VII},
  pages={557--577},
  year={2022},
  organization={Springer}
}
```
