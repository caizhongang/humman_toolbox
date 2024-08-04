# HuMMan Release v1.0: 3D Vision Subset

HuMMan v1.0: 3D Vision Subset consists of 340 subjects, 247 actions and 907 sequences. 
Color videos, depth images, masks (computed with background), SMPL parameters, and camera 
parameters are provided. It is worth noting that data captured with a mobile sensor (iPhone)
are also included. This subset is ideal for 3D vision researchers to study dynamic humans
with commercial depth sensors.  

### Downloads

Color videos (Kinect):
- Part 1: [Aliyun]() (~ GB) 
- Part 2: [Aliyun]() (~ GB)

Depth images (Kinect): 
- Part 1: [Aliyun]() (~ GB) 
- Part 2: [Aliyun]() (~ GB)
- Part 3: [Aliyun]() (~ GB) 
- Part 4: [Aliyun]() (~ GB)
- Part 5: [Aliyun]() (~ GB) 
- Part 6: [Aliyun]() (~ GB)
- Part 7: [Aliyun]() (~ GB) 
- Part 8: [Aliyun]() (~ GB)
- Part 9: [Aliyun]() (~ GB) 
- Part 10: [Aliyun]() (~ GB)

Masks: [Aliyun]() (~ GB)

Color videos (iPhone): [Aliyun]() (~ GB) 

Depth images (iPhone): [Aliyun]() (~ GB)

SMPL parameters: [Aliyun]() (~ GB)

Camera parameters (world2cam): [Aliyun]() (~ GB)

Background: [Aliyun]() (~ GB)

### Data Structure
Please download the `.7z` files and place in the same directory, note that you may not need all of them.
```text
humman_release_v1.0_recon/   
├── point_kinect_color_part_01.7z
├── point_kinect_color_part_02.7z
├── point_kinect_depth_part_01.7z
├── point_kinect_depth_part_02.7z
├── point_kinect_depth_part_03.7z
├── point_kinect_depth_part_04.7z
├── point_kinect_depth_part_05.7z
├── point_kinect_depth_part_06.7z
├── point_kinect_depth_part_07.7z
├── point_kinect_depth_part_08.7z
├── point_kinect_depth_part_09.7z
├── point_kinect_depth_part_10.7z
├── point_kinect_depth_part_11.7z
├── point_kinect_depth_part_12.7z
├── point_kinect_mask.7z
├── point_iphone_color.7z
├── point_iphone_depth.7z
├── point_smpl_params.7z 
├── point_cameras.7z
└── point_background.7z
```
Then decompress them:
```bash
unzip "*.7z" (TODO!)
```
The file structure should look like this:
```text
humman_release_v1.0_point/   
└── pxxxxxx_axxxxxx/  
    ├── background/
    │   ├── kinect_color_000.jpg (uint8, 3 channel)
    │   ...
    │   ├── kinect_color_009.jpg
    │   ├── kinect_depth_000.png (uint16, 1 channel)
    │   ...
    │   └── kinect_depth_009.png
    │
    ├── iphone_color/
    │   └── iphone.mp4
    │
    ├── iphone_depth/
    │   └── iphone/
    │       ├── 000000.png (uint16, 1 channel)
    │       ├── 000001.png
    │       ...
    │       └── xxxxxx.png
    │
    ├── kinect_color/
    │   ├── kinect_000.mp4
    │   ...
    │   └── kinect_009.mp4
    │
    ├── kinect_depth/
    │   ├── kinect_000/
    │   ...
    │   └── kinect_009/
    │       ├── 000000.png (uint16, 1 channel)
    │       ├── 000001.png
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
    ├── cameras.json
    │
    └── smpl_params.npz
```


### Split Color Videos
To save storage, the color (RGB) data comes in compressed video format (.mp4).  
We provide a tool to split color videos into color images.
However, please note that spliting the videos leads to approximately **100x** larger total file size.
```bash
python tools/video_splitter.py [--root_dir] [--seq_name]
```
- root_dir (str): root directory in which data is stored. Required.
- seq_name (str): sequence name, in the format 'pxxxxxx_axxxxxx'. Optional, all sequences will be processed if not specified.

For example,
```bash
python tools/video_splitter.py --root_dir /home/user/humman_release_v1.0_point/ --seq_name p000438_a000040
```


### Data Loading

#### kinect_color/ and iphone_color/ (TODO!)
To save
```python
import cv2
color_bgr = cv2.imread('/path/to/xxxxxxx.png')
color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)  # if RGB images are used
```

#### kinect_depth/ and iphone_depth/
To read the depth maps
```python
import cv2
depth_image = cv2.imread('/path/to/xxxxxxx.png', cv2.IMREAD_UNCHANGED)
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
    'betas': np.array of shape (n, 10),
    'body_pose': np.array of shape (n, 69),
    'global_orient': np.array of shape (n, 3),
    'transl': np.array of shape (n, 3)
}
```
where `n` is the number of frames.

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

To convert the depth maps to point clouds in camera coordinate system (TODO!)
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

#### Run RGB-D Visualizer
```bash
python tools/visualizer <root_dir> <seq_name> <kinect_id> <frame_id> \
  [--visualize_mask] [--visualize_smpl] [--smpl_model_path]
```
- root_dir (str): root directory in which data is stored.
- seq_name (str): sequence name, in the format 'pxxxxxx_axxxxxx'.
- kinect_id (int): Kinect ID. Available range is [0, 9].
- frame_id (int): frame ID. Available range varies for different sequences.
- visualize_mask (bool, optional): whether to overlay mask on color image. Defaults to False.
- visualize_mask_manual (bool, optional): whether to overlay manually annotated mask on color image. Defaults to False.
- visualize_smpl (bool, optional): whether to overlay SMPL vertices on color image. Defaults to False.
- smpl_model_path (str, optional): directory in which SMPL body models are stored.

Example:
```bash
python tools/visualizer /home/user/humman_release_v1.0_recon p000455_a000986 0 0 \
  --visualize_mask_manual --visualize_smpl --smpl_model_path /home/user/body_models/
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
- visualize_smpl (bool, optional): whether to visualize SMPL 3D mesh model.
- smpl_model_path (str, optional): directory in which SMPL body models are stored.

Example:
```bash
python tools/visualizer /home/user/humman_release_v1.0_recon p000455_a000986 0 0 \
  --visualize_smpl --smpl_model_path /home/user/body_models/
```


