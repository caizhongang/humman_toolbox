# HuMMan Release v1.0: Reconstruction Subset (HuMMan-Recon)

HuMMan v1.0: Reconstruction Subset (HuMMan-Recon) consists of 153 subjects and 339 sequences. 
Color images, masks (via matting), SMPL parameters, and camera parameters are 
provided. It is a challenging dataset for its collection of diverse subject 
appearance and expressive actions. Moreover, it unleashes the potential to 
benchmark reconstruction algorithms under realistic settings with commercial 
sensors, dynamic subjects, and computer vision-powered automatic annotations.

### Installation
To use our visulization tools, relevant python packages need to be installed.
```bash
conda create -n humman python=3.9 -y
conda activate humman
pip install torch==1.12.1 opencv-python==4.10.0.84 smplx==0.1.28 chumpy==0.70 trimesh==4.4.3 tqdm==4.66.4 open3d==0.14.1 numpy==1.23.1
```
It is also highly recommended to install `openxlab` package to facilitate file downloading.
```bash
pip install openxlab
```

### Downloads

#### Option 1: OpenXLab

HuMMan-Recon is currently hosted on [OpenXLab](https://openxlab.org.cn/datasets/OpenXDLab/HuMMan/tree/main/humman_release_v1.0_recon).
We recommend download files using [CLI tools](https://openxlab.org.cn/datasets/OpenXDLab/HuMMan/cli/main):
```bash
openxlab dataset download --dataset-repo OpenXDLab/HuMMan --source-path /humman_release_v1.0_recon --target-path /home/user/
```

You can selectively download files that you need, for example:
```bash
openxlab dataset download --dataset-repo OpenXDLab/HuMMan --source-path /humman_release_v1.0_recon/recon_kinect_color_part_1.zip --target-path /home/user/humman_release_v1.0_recon
```

#### Option 2: OneDrive

We have backed-up all files on OneDrive (except for depth images).

Color images:
- Part 1: [OneDrive(CN)](https://pjlab-my.sharepoint.cn/:u:/g/personal/openmmlab_pjlab_org_cn/Ed6CM2tKVmRGiqcCyIcd4rABYy8kcf_tIbDU4nmuM4Zi1Q?e=yrC0pT) 
(~84 GB)
- Part 2: [OneDrive(CN)](https://pjlab-my.sharepoint.cn/:u:/g/personal/openmmlab_pjlab_org_cn/EY1aLq6U9ppJsYuMy_y30yABGhrepLtgk4KQ_9Xl_O3heQ?e=7cTTvM) 
(~73 GB)
- Part 3: [OneDrive(CN)](https://pjlab-my.sharepoint.cn/:u:/g/personal/openmmlab_pjlab_org_cn/Edd6RwJ-9VJCpV99BKXp0SgBNvvtLdZmJEOvrFv_K8Aj1w?e=XkjLTu) 
(~84 GB)

Masks:
- Manually annotated for color images in the test split only:
[OneDrive(CN)](https://pjlab-my.sharepoint.cn/:u:/g/personal/openmmlab_pjlab_org_cn/ESrcqBN8SltMq-dk1OW-XVYBUiU8TijQjWGRm107o1RsYg?e=AttU91)
(~32 MB)
- Generated via matting for color images in all splits: 
[OneDrive(CN)](https://pjlab-my.sharepoint.cn/:u:/g/personal/openmmlab_pjlab_org_cn/ERVZ8yDijvZItS7ptjT3JiIB4dE0QJ-8802bWdWGidBl1g?e=DnzSm1) 
(~2.1 GB)

SMPL parameters: 
[OneDrive(CN)](https://pjlab-my.sharepoint.cn/:u:/g/personal/openmmlab_pjlab_org_cn/EbYzP89JCppHnGWn6WGyZUcBt9ZSRIZdJIm--cJVmu7XKw?e=Bq4HPp)
(~7.6 MB)

Camera parameters (world2cam): 
[OneDrive(CN)](https://pjlab-my.sharepoint.cn/:u:/g/personal/openmmlab_pjlab_org_cn/EZyvqo2fKhdBmyLj6Tchf8MB_dUr2cneVwCdD29l6nyP2A?e=GsLBqF)
(~1.3 MB). 

Textured meshes: 
[OneDrive(CN)](https://pjlab-my.sharepoint.cn/:u:/g/personal/openmmlab_pjlab_org_cn/EYgbLNilus1Auut9N-qs5rkBZ-NuNYWG5ml-tRxTlgeohA?e=Sdaogm)
(~22 GB)

Suggested splits:
[train](https://caizhongang.github.io/projects/HuMMan/splits/train.txt) and
[test](https://caizhongang.github.io/projects/HuMMan/splits/test.txt).


### Data Structure
Please download the .zip files and place in the same directory, note that you may not need all of them.
```text
humman_release_v1.0_recon/   
├── recon_kinect_color_part_1.zip
├── recon_kinect_color_part_2.zip
├── recon_kinect_color_part_3.zip
├── recon_kinect_mask_manual.zip
├── recon_kinect_mask.zip
├── recon_smpl_params.zip 
├── recon_cameras.zip
├── recon_textured_meshes.zip
└── recon_kinect_depth.zip
```
Then decompress them:
```bash
unzip "*.zip"
```
The file structure should look like this:
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
    ├── kinect_mask_manual/
    │   ├── kinect_000/
    │   ...
    │   └── kinect_009/
    │       ├── 000000.png (uint8, 1 channel)
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

#### kinect_mask/ or kinect_mask_manual/
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

R, T form **world2cam** transformation.
The world coordinate system is the same as kinect_color_000 coordinate system. 

Each .json consists of parameters for 10 color cameras and 10 depth cameras:
```text
{
    "kinect_color_000": {
        "K": np.array of shape (3,3),
        "R": np.array of shape (3,3),
        "T": np.array of shape (3,)
    },
    "kinect_depth_000": {
        "K": np.array of shape (3,3),
        "R": np.array of shape (3,3),
        "T": np.array of shape (3,)
    },
    ...
    "kinect_color_009": {...}
    "kinect_depth_009": {...}
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
depth_image = o3d.geometry.Image(depth_image)
open3d_point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
    depth_image, open3d_camera.intrinsic, depth_trunc=5.0)
point_cloud = np.array(open3d_point_cloud.points)
```

### Visualization
We provide a simple 2D visualization tool for color images, masks, and SMPL vertices,
and a simple 3D visualization tool for point clouds (from depth images) and SMPL mesh models.

#### Run 2D Visualizer
```bash
python tools/visualizer.py <root_dir> <seq_name> <kinect_id> <frame_id> \
  [--visualize_mask] [--visualize_smpl] [--smpl_model_path]
```
- root_dir (str): root directory in which data is stored.
- seq_name (str): sequence name, in the format 'pxxxxxx_axxxxxx'.
- kinect_id (int): Kinect ID. Available range is [0, 9].
- frame_id (int): frame ID. Available range varies for different sequences.
- visualize_mask (flag): whether to overlay mask on color image.
- visualize_mask_manual (flag): whether to overlay manually annotated mask on color image.
- visualize_smpl (flag): whether to overlay SMPL vertices on color image.
- smpl_model_path (str, optional): directory in which SMPL body models are stored. Defaults to /home/user/body_models/.

Example:
```bash
python tools/visualizer.py /home/user/humman_release_v1.0_recon p000455_a000986 0 0 \
  --visualize_mask_manual --visualize_smpl --smpl_model_path /home/user/body_models/
```

Note that the SMPL model path should consist the following structure:
```text
body_models/   
└── smpl/  
    └── SMPL_NEUTRAL.pkl
```
`SMPL_NEUTRAL.pkl` may be downloaded from the [official website](https://smpl.is.tue.mpg.de/). 


#### Run 3D Visualizer
```bash
python tools/visualizer_3d.py <root_dir> <seq_name> <kinect_id> <frame_id> \
  [--visualize_smpl] [--smpl_model_path]
```
- root_dir (str): root directory in which data is stored.
- seq_name (str): sequence name, in the format 'pxxxxxx_axxxxxx'.
- kinect_id (int): Kinect ID. Available range is [0, 9].
- frame_id (int): frame ID. Available range varies for different sequences.
- visualize_smpl (bool, optional): whether to visualize SMPL 3D mesh model.
- smpl_model_path (str, optional): directory in which SMPL body models are stored.

Example:
```bash
python tools/visualizer_3d.py /home/user/humman_release_v1.0_recon p000455_a000986 0 0 \
  --visualize_smpl --smpl_model_path /home/user/body_models/
```


