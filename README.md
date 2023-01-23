# Toolbox for HuMMan Dataset

Please visit our [Homepage](https://caizhongang.github.io/projects/HuMMan/) for more details.      

## Updates
- [2023-01] Release of HuMMan v1.0: Reconstruction Subset
- [2022-10] We presented HuMMan as an oral paper at ECCV'22 (Tel Aviv, Israel)
- [2022-08] Release of HuMMan v0.1 (no longer available, please use v1.0)

## HuMMan Release v1.0: Reconstruction Subset

HuMMan v1.0: Reconstruction Subset consists of 153 subjects and 339 sequences. 
Color images, masks (via matting), SMPL parameters, and camera parameters are 
provided. It is a challenging dataset for its collection of diverse subject 
appearance and expressive actions. Moreover, it unleashes the potential to 
benchmark reconstruction algorithms under realistic settings with commercial 
sensors, dynamic subjects, and computer vision-powered automatic annotations.

### Downloads

- Part 1: [Aliyun](https://openxdlab.oss-cn-shanghai.aliyuncs.com/HuMMan/humman_release_v1.0_recon/recon_part_1.zip) 
or [OneDrive(CN)](https://pjlab-my.sharepoint.cn/:u:/g/personal/openmmlab_pjlab_org_cn/EY5YzfZkY3BFsUcnAEsAtl0Bb3s9E4xTzNjncBWAyxXpwQ?e=92xOu7) (~80 GB)
- Part 2: [Aliyun](https://openxdlab.oss-cn-shanghai.aliyuncs.com/HuMMan/humman_release_v1.0_recon/recon_part_2.zip) 
or [OneDrive(CN)](https://pjlab-my.sharepoint.cn/:u:/g/personal/openmmlab_pjlab_org_cn/EZFK-Eykx7ZAgdydMQoqw7EBPZUNLHVIXT2SmTEZLw9VmA?e=oiZnpd) (~80 GB)
- Part 3: [Aliyun](https://openxdlab.oss-cn-shanghai.aliyuncs.com/HuMMan/humman_release_v1.0_recon/recon_part_3.zip) 
or [OneDrive(CN)](https://pjlab-my.sharepoint.cn/:u:/g/personal/openmmlab_pjlab_org_cn/Ed59pXLm2DhMoj0rolBIfh8BYHKIX3uxuGeeThJ03GaRZQ?e=5hYSRA) (~80 GB)

### Data Structure

```text
humman_release_v1.0_recon   
└── pxxxxxx_axxxxxx  
    ├── kinect_color
    │   ├── kinect_000
    │   ...
    │   └── kinect_009
    │       ├── 000000.png
    │       ├── 000006.png
    │       ...
    │       └── xxxxxx.png
    │
    ├── kinect_mask
    │   ├── kinect_000
    │   ...
    │   └── kinect_009
    │       ├── 000000.png
    │       ├── 000006.png
    │       ...
    │       └── xxxxxx.png
    │
    ├── smpl_params
    │   ├── 000000.npz
    │   ├── 000006.npz
    │   ...
    │   └── xxxxxx.npz
    │
    ├── cameras.json
    │
    └── textured_meshes (optional)
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
- \* indicates that there be multiple png for one .obj file.

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

### Visualization
We provide a simple visualization tool for color images, masks, and SMPL vertices.

#### Installation
The tool does not require specific version of dependency packages. 
The following is for reference only. 
```bash
conda create -n humman python=3.8 -y
conda activate humman
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install opencv-python smplx
```

#### Run Visualizer
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

Note that the SMPL model path should consists the following structure:
```text
body_models/   
└── smpl  
    └── SMPL_NEUTRAL.pkl
```
`SMPL_NEUTRAL.pkl` may be downloaded from [here](https://github.com/classner/up/blob/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl). 
Renaming from `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` to `SMPL_NEUTRAL.pkl` is needed.