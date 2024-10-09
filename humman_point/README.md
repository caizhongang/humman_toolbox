# HuMMan Release v1.0: 3D Vision Subset (HuMMan-Point)

HuMMan v1.0: 3D Vision Subset (HuMMan-Point) consists of 340 subjects, 247 actions and 907 sequences. 
Color videos, depth images, masks (computed with background), SMPL parameters, and camera 
parameters are provided. It is worth noting that data captured with a mobile sensor (iPhone)
are also included. This subset is ideal for 3D vision researchers to study dynamic humans
with commercial RGB-D sensors.  

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

HuMMan-Point is currently hosted on [OpenXLab](https://openxlab.org.cn/datasets/OpenXDLab/HuMMan/tree/main/humman_release_v1.0_point).
We recommend download files using [CLI tools](https://openxlab.org.cn/datasets/OpenXDLab/HuMMan/cli/main):
```bash
openxlab dataset download --dataset-repo OpenXDLab/HuMMan --source-path /humman_release_v1.0_point --target-path /home/user/
```

You can selectively download files that you need, for example:
```bash
openxlab dataset download --dataset-repo OpenXDLab/HuMMan --source-path /humman_release_v1.0_point/point_iphone_color.7z --target-path /home/user/humman_release_v1.0_point/
```

#### Option 2: Hugging Face

HuMMan-Point is also hosted on [Hugging Face](https://huggingface.co/datasets/caizhongang/HuMMan/tree/main/humman_release_v1.0_point).
Hugging Face uses `git-lfs` to manage large files.

Please make sure you have [git-lfs](https://git-lfs.com) installed. Then, follow the instructions below:
```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/caizhongang/HuMMan  # do not pull any large files yet
cd HuMMan
```

You may pull all files in HuMMan-Point:
```
git lfs pull --include "humman_release_v1.0_point/*"
```

Similarly, you can also selectively download files that you need, for example:
```bash
git lfs pull --include "humman_release_v1.0_point/point_iphone_color.7z"
```


### Data Structure
Please download the `.7z` files and place in the same directory, note that you may not need all files.
```text
humman_release_v1.0_recon/   
├── point_background.7z
├── point_cameras.7z
├── point_iphone_color.7z
├── point_iphone_depth.7z
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
├── point_kinect_mask.7z
└── point_smpl_params.7z 
```
Then decompress them:
```bash
7z x "*.7z"
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

#### kinect_color/ and iphone_color/ 
To read the video directly:
```python
cap = cv2.VideoCapture('/path/to/xxxxxxx.mp4')
color_rgb_list = []
while cap.isOpened():
    success, color_bgr = cap.read()
    if not success:
        break
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB) # if RGB images are used
    color_rgb_list.append(color_rgb)
cap.release()
```
Otherwise, to read from frames split from the videos (see `video_splitter.py` above):
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

#### kinect_depth/ or iphone_depth/

To read the depth maps 
```python
import cv2
depth_image = cv2.imread('/path/to/xxxxxxx.png', cv2.IMREAD_UNCHANGED)
```

To convert the depth maps to point clouds in **depth** camera coordinate system
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

To convert the depth maps to color point clouds in **color** camera coordinate system, please take the `create_rgbd` function in `visualizer_rgbd.py` as a reference. Note for kinects, the color and depth cameras are not perfectly aligned in terms of camera coordiante systems.

### Visualization
We provide a simple 3D visualization tool for point clouds (from depth images) and SMPL mesh models. If `frame_id` is specified, an interactive visualizer is used to visualize oen frame of RGB-D data with an interactive viewer (adjustable camera pose); otherwise a sequence visualizer is used to visualize the entire RGB-D video with a pre-set virtual camera.

#### Run RGB-D Visualizer
```bash
python tools/visualizer_rgbd.py <root_dir> <seq_name> <device_name> [frame_id] \
  [virtual_cam] [video_save_path] [--visualize_smpl] [--smpl_model_path]
```
- root_dir (str): root directory in which data is stored.
- seq_name (str): sequence name, in the format 'pxxxxxx_axxxxxx'.
- device_name (str): device name. 'kinect_000' to 'kinect_009' or 'iphone'.
- frame_id (int): frame ID. If not specified, the entire video will be visualized.
- virtual_cam (str, optional): virtual camera pose. Required for visualizing the entire video. Defaults to assets/virtual_cam.json.
- video_save_path (str, optional): path to save the visualization video. If not specified, it will be ./{seq_name}-{device_name}.mp4
- visualize_smpl (flag): whether to visualize SMPL 3D mesh model. 
- smpl_model_path (str, optional): directory in which SMPL body models are stored. Defaults to /home/user/body_models/.

Example 1: use the interactive visualizer to visualize frame '0' of 'kinect_000':
```bash
python tools/visualizer_rgbd.py /home/user/humman_release_v1.0_point p001110_a001425 kinect_000 --frame_id 0 
```

Example 2: use the sequence visualizer to visualize the entire 'iphone' video:
```bash
python tools/visualizer_rgbd.py /home/user/humman_release_v1.0_point p001110_a001425 iphone --virtual_cam assets/virtual_cam_iphone.json 
```


Note that the SMPL model path should consist the following structure:
```text
body_models/   
└── smpl/  
    └── SMPL_NEUTRAL.pkl
```
`SMPL_NEUTRAL.pkl` may be downloaded from the [official website](https://smpl.is.tue.mpg.de/). 
