# Toolbox for HuMMan Dataset

Please visit our [Homepage](https://caizhongang.github.io/projects/HuMMan/) for more details.      

## Updates
- [2024-10-02] HuMMan is now available on [HuggingFace](https://huggingface.co/datasets/caizhongang/HuMMan)!
- [2024-09-10] Release of HuMMan v1.0: Motion Generation Subset (HuMMan-MoGen)
- [2024-08-29] Release of HuMMan v1.0: 3D Vision Subset (HuMMan-Point)
- [2024-08-29] We have changed our data host! All download instructions have been updated
- [2024-07-27] HuMMan-Recon: Release of depth maps for the Reconstruction Subset
- [2023-04-25] HuMMan-Recon: Release of manually annotated masks for color images in the test split
- [2023-02-27] HuMMan-Recon: Downloads are organized by modalities, links have been updated
- [2023-01-23] HuMMan-Recon: Release of textured meshes for the Reconstruction Subset, and a toolbox
- [2023-01-23] HuMMan-Recon: Minor fixes on the mask data, download links have been updated
- [2023-01-11] Release of HuMMan v1.0: Reconstruction Subset (HuMMan-Recon)
- [2022-10-27] We presented HuMMan as an oral paper at ECCV'22 (Tel Aviv, Israel)
- [2022-08] Release of HuMMan v0.1 (no longer available, please use v1.0)

## Datasets

- [HuMMan-Recon](humman_recon/): HuMMan v1.0: Reconstruction Subset
- [HuMMan-Point](humman_point/): HuMMan v1.0: 3D Vision Subset
- [HuMMan-MoGen](humman_mogen/): HuMMan v1.0: Motion Generation Subset


## Citation
Please cite our work if you use our datasets (`HuMMan-Recon`, `HuMMan-Point`, or `HuMMan-MoGen`) in your research.
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

Please also cite *FineMoGen* if you use the `HuMMan-MoGen` subset.
```text
@article{zhang2023finemogen,
  title   =   {FineMoGen: Fine-Grained Spatio-Temporal Motion Generation and Editing}, 
  author  =   {Zhang, Mingyuan and Li, Huirong and Cai, Zhongang and Ren, Jiawei and Yang, Lei and Liu, Ziwei},
  year    =   {2023},
  journal =   {NeurIPS},
}
```