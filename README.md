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
    
*: may have multiple png
```

#### smpl_parms/
```text
Each .npz consists of the following:
{
    'betas': np.array of shape (10,),
    'body_pose': np.array of shape (69,),
    'global_orient': np.array of shape (3,),
    'transl': np.array of shape (3,)
}

Note:
SMPL parameters are in the world coordinate system.
The world coordinate system is the same as kinect_color_000 coordinate system. 
```

#### cameras.json
```text
Each .json consists of the following:
{
    "kinect_color_000": {
        "K": np.array of shape (3,3),
        "R": np.array of shape (3,3),
        "T": np.array of shape (3,)
    },
    ...
    "kinect_color_009": {...}
}

Note:
Cameras use the OpenCV pinhole camera model.
K: intrinsic matrix
R: rotation matrix
T: translation

R, T form world2cam transformation.
The world coordinate system is the same as kinect_color_000 coordinate system. 
```