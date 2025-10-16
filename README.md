#  Enhancing Lightweight Image Super-Resolution through Frequency-Enhanced Mamba Networks


> **Abstract:** Image Super-Resolution (SR) aims to enhance low-resolution images to high-resolution counterparts with superior quality. Despite notable advancements using deep learning, maintaining computational efficiency while capturing long-range dependencies remains challenging. Inspired by state space models, we introduce the Frequency Enhanced Mamba Network (FEMN), designed to integrate global and local image features across frequency and spatial domains. Our approach incorporates a Frequency Feature Extraction Module (FFEM) for adaptive frequency cue extraction and a Localized Spatial Aggregation Module (LSAM) for enhanced spatial variation representation. Extensive experiments on standard benchmarks demonstrate FEMN's superior performance and efficiency, achieving state-of-the-art reconstruction quality while maintaining low computational complexity.

## Environment & Dependencies

To ensure seamless execution of our project, the following dependencies are required:

* Python == 3.12.3
* Pytorch == 2.5.1
* cudatoolkit == 11.0.221

We export our conda virtual environment as environment.yaml. You can use the following command to create the environment.

```bash
conda env create -f environment.yaml
```

This ensures all dependencies are correctly installed, allowing you to focus on running and experimenting with the code.

## Datasets

The datasets used in our training and testing are orgnized as follows:

| Task                           | Training Set                                                                                                                                                                                                                                                                                                                                                                                                                                    |                                                                              Testing Set                                                                              
|:-------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------: |
| lightweight Image SR           | LightSR:[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)<br />                                                                                                                                                                                                                                                                                                                                                                                |               Set5 + Set14 + BSD100 + Urban100 + Manga109 [[download](https://drive.google.com/file/d/1n-7pmwjP0isZBK7w3tx2y8CTastlABx1/view?usp=sharing)]               | 
| Gaussian Color Image Denoising | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) +  [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) + [BSD400](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) + [WED](http://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar) <br />[complete DFWB_RGB [download](https://drive.google.com/file/d/1jPgG_URDQZ4kyXaMMXJ8AZ8jEErCdKuM/view?usp=share_link)] |CBSD68 + Kodak24 + McMaster [[download](https://drive.google.com/file/d/1baLpOjNlTCNbREUDAZf9Lso6YCeUOQER/view?usp=sharing)]                 | | [GoPro](https://drive.google.com/file/d/1abXSfeRGrzj2mQ2n2vIBHtObU6vXvr7C/view) + [HIDE](https://drive.google.com/file/d/1XRomKYJF1H92g1EuD06pCQe4o6HlwB7A/view?usp=sharing) |  [Download](https://drive.google.com/drive/folders/1cA3PgLYGTW_ofC8wPBR3DUlhpvuRDwuw?usp=sharing)  |
| Image Dehazing                 | Indoor & Outdoor:[RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-standard?authuser=0) (including 13990 indoor and 313950 outdoor images)<br />                                                                                                                                 |                                 SOTS [[download](https://drive.google.com/drive/folders/1oaQSpdYHxEv-nMOB7yCLKfw2NDCJVtrx)]                                 | 

### Training and Testing Commands on Super-Resolution

```bash
# Training commands for scale of x2, x3, x4 lightSR 
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=8888 basicsr/train.py -opt options/train/train_FEMamba_lightSR_x2.yml --launcher pytorch
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=8888 basicsr/train.py -opt options/train/train_FEMamba_lightSR_x3.yml --launcher pytorch
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=8888 basicsr/train.py -opt options/train/train_FEMamba_lightSR_x4.yml --launcher pytorch

# Testing commands for scale of x2, x3, x4 lightSR 
python basicsr/test.py -opt options/test/test_FEMamba_lightSR_x2.yml
python basicsr/test.py -opt options/test/test_FEMamba_lightSR_x3.yml
python basicsr/test.py -opt options/test/test_FEMamba_lightSR_x4.yml
```
### Training and Testing Commands on Color Image Denoising
```bash
# Training commands for Color Denoising with sigma=50
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=8888 basicsr/trainF.py -opt options/train/train_FEMamba_CDN_s50.yml --launcher pytorch

# Testing commands for Color Denoising with sigma=50
python basicsr/test.py -opt options/test/test_FEMamba_CDN_s50.yml
```

### Training and Testing Commands on Image Dehazing
```bash
# Training commands for Image Dehazing
torchrun --nproc_per_node=8 --master_port=8888 realDenoising/basicsr/trainF.py -opt realDenoising/options/train/train_FEMamba_ITS.yml --launcher pytorch

# Testing commands for Image Dehazing
python realDenoising/basicsr/test.py -opt realDenoising/options/test/test_FEMamba_ITS.yml

```

Cautions: torchrun is only available for pytorch>=1.9.0. If you do not want to use torchrun for training, you can replace it with `python -m torch.distributed.launch` for training.


## Acknowledgement

This code and README is based on [MambaIR](https://github.com/csguoh/MambaIR/) and [MaIR](https://github.com/XLearning-SCU/2025-CVPR-MaIR). Many thanks for their awesome work.
