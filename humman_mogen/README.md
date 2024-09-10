# HuMMan Release v1.0: Motion Generation Subset

HuMMan v1.0: Motion Generation Subset (HuMMan-MoGen) consists of 160 actions (320 after mirrored), 179 subjects, 6264 motion sequences and 112112 fine-grained text descriptions. 
This dataset is designed to facilitate a large-scale study on the fine-grained motion generation task. 
It features temporal (by stage) and spatial (by part) text annotation of each SMPL motion sequence. 
Specifically, each motion sequence is divided into multiple standard action phases.
For each phase, it is not only annotated with an overall description, but seven more detailed annotations to 
describe the head, torso, left arm, right arm, left leg, right leg, and trajectory of the pelvis joint. 
Please see our [demo video](https://youtu.be/xYgii9vqG08) for a few examples.

### Downloads

The dataset can be downloaded from [OpenXLab](https://openxlab.org.cn/datasets/OpenXDLab/HuMMan/tree/main/humman_release_v1.0_mogen):
- SMPL parameters and annotations (~300 MB)
- Visualization videos (optional, ~8.8 GB)


### Data Structure
Please download the .zip files and place in the same directory, note that you may not need all of them.
```text
humman_release_v1.0_mogen/   
├── data.zip
└── visualizations.zip (optional)
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

Example of annotation file: 
```text
[
    {
        "start_frame": 0,
        "end_frame": 21,
        "num_frames": 21,
        "all": "Stand with legs open.",
        "head": "Maintain a neutral position, facing forward.",
        "stem": "Keep the spine straight and aligned.",
        "left_arm": "Keep the left arm by the side of the body.",
        "right_arm": "Keep the right arm by the side of the body.",
        "left_leg": "Open the legs apart.",
        "right_leg": "Open the legs apart.",
        "pelvis": "The pelvis remains stable and centered in its natural position."
    },
    {
        "start_frame": 21,
        "end_frame": 66,
        "num_frames": 45,
        "all": "Take several steps to the left with your elbows bent and arms swung back and forth alternatively,  
                then lift the right leg, and swing left arm to the front, keep the left elbow close to the right knee.",
        "head": "Maintain a neutral position, facing forward.",
        "stem": "Keep the spine straight and aligned throughout the motion.",
        "left_arm": "Swing the left arm back and forth alternately with the right arm. 
                    Then, swing the left arm to the front and keep the left elbow close to the right knee.",
        "right_arm": "Swing the right arm back and forth alternately with the left arm. 
                    Then, swing the right arm down.",
        "left_leg": "Take several steps to the left.",
        "right_leg": "Take several steps to the left and lift the right leg.",
        "pelvis": "The pelvis shifts to the left during the steps."
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
The visualized videos (`visualizations.zip`) of SMPL mesh models can be directly downloaded.
In case you'd like to run it yourself, we also provide the simple 3D visualization tool. 

#### Installation
```bash
conda create -n humman python=3.8 -y
conda activate humman
pip install numpy==1.23.0
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge -y
pip install opencv-python smplx chumpy trimesh==4.4.7
pip install open3d==0.16.0  # additional package for 3D visualization. Version 0.16.0 is recommended.
```

#### Run 3D Visualizer
```bash
python tools/visualizer_3d.py <root_dir> <seq_name> <video_dir> [--draw_ground] [--draw_axis] [--smpl_model_path]
```
- root_dir (str): root directory in which smpl sequences (.npz) are stored.
- seq_name (str): sequence name, in the format 'pxxxxxx_axxxxxx' or 'pxxxxxx_rxxxxxx'. Input '*' to visualize all sequences under the root directory. 
- video_dir (str): directory to save rendered videos.
- draw_ground (bool, optional): whether to visualize a ground plane.
- draw_axis (bool, optional): whether to visualize the xyz axis.
- smpl_model_path (str, optional): directory in which SMPL body models are stored. You can download from the [official website](https://smpl.is.tue.mpg.de/).

Example:
```bash
python tools/visualizer_3d.py /home/user/humman_release_v1.0_mogen/smpl_on_ground p000459_a000011 /home/user/humman_release_v1.0_mogen/visualizations \
  --draw_ground --draw_axis --smpl_model_path /home/user/body_models/
```

Example of SMPL body model directory (smpl neutral model is used in our 3D Visualizer):
```text
body_models/
├── smpl
│   ├── SMPL_NEUTRAL.pkl
│   ├── SMPL_FEMALE.pkl
│   ├── SMPL_MALE.pkl
│   └── ......
├── smplx
└── ......


