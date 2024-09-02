# HuMMan Release v1.0: Motion Generation Subset

HuMMan v1.0: Mogen Subset consists of 160 actions, 179 subjects and 6264 motion sequences. 
SMPL parameters, stage divisions, overall & bodypart annotations and video visualizations are provided. 

### Downloads

SMPL parameters and Annotations (~300 MB)

Visualizations (mp4) (~8.8 GB)

[OpenXLab Link](https://openxlab.org.cn/datasets/OpenXDLab/HuMMan/tree/main/humman_release_v1.0_mogen) 


### Data Structure
Please download the .zip files and place in the same directory, note that you may not need all of them.
```text
humman_release_v1.0_mogen/   
├── data.zip
└── visualizations.zip
```
Then decompress them:
```bash
unzip "*.zip"
```
The file structure should look like this:
```text
humman_release_v1.0_mogen/
├── smpl_on_ground
│   ├── p000459_a000011.npz
│   ├── p000460_a000011.npz
│   ├── p000459_r000011.npz
│   └── ......
├── annotations
│   ├── p000459_a000011.json
│   ├── p000460_a000011.json
│   ├── p000459_r000011.json
│   └── ......
└── visualizations
    ├── p000459_a000011.mp4
    ├── p000460_a000011.mp4
    ├── p000459_r000011.mp4
    └── ......
    
```
Files are named in the format of "PersonID_ActionID", representing the ID of the human subject and the action performed. Action named 'rxxxxxx' is a mirror of 'axxxxxx'. Motion sequences are stored in npz files under "./smpl_on_ground", in SMPL format (23 joints). Annotation .json files are under "./annotations". Visualization mp4 videos are under "./visualizations".


#### smpl_parms/
Each .npz consists of the following:
```text
{
    'betas': np.array of shape (T, 10),
    'body_pose': np.array of shape (T, 69),
    'global_orient': np.array of shape (T, 3),
    'transl': np.array of shape (T, 3)
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

#### Annotations
Each .json annotation file consists of a list of dictionaries, each describing a substage of the motion sequence, with start frame (included) & end frame (excluded), length, and overall & bodypart text descriptions of the substage.
```text
[
    {
        "start_frame": 0,
        "end_frame": 78,
        "num_frames": 78,
        "all": "string", 
        "head": "string", 
        "stem": "string", 
        "left_arm": "string", 
        "right_arm": "string", 
        "left_leg": "string", 
        "right_leg": "string", 
        "pelvis": "string"
    },
    {
        ...
    },
    ...
]
```

To read:
```python
import json
with open('/path/to/xxxxxx.json', 'r') as f:
    stages = json.load(f)
```

### Visualization
We provide a simple 3D visualization tool for SMPL mesh models.

#### Installation
The tool does not require specific version of dependency packages. 
The following is for reference only. 
```bash
conda create -n humman python=3.8 -y
conda activate humman
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge -y
pip install opencv-python smplx chumpy trimesh
pip install open3d==0.16.0  # additional package for 3D visualization. Version 0.16.0 is recommended.
```

#### Run 3D Visualizer
```bash
python tools/visualizer_3d <root_dir> <seq_name> <video_dir> [--draw_ground] [--draw_axis] [--smpl_model_path]
```
- root_dir (str): root directory in which smpl sequences (.npz) are stored.
- seq_name (str): sequence name, in the format 'pxxxxxx_axxxxxx' or 'pxxxxxx_rxxxxxx'. Input '*' to visualize all sequences under the root directory. 
- video_dir (str): directory to save rendered videos.
- draw_ground (bool, optional): whether to visualize a ground plane.
- draw_axis (bool, optional): whether to visualize the xyz axis.
- smpl_model_path (str, optional): directory in which SMPL body models are stored. 

Example:
```bash
python tools/visualizer_3d /home/user/humman_release_v1.0_mogen/smpl_on_ground p000459_a000011 /home/user/humman_release_v1.0_mogen/visualizations \
  --draw_ground --draw_axis --smpl_model_path /home/user/body_models/
```


