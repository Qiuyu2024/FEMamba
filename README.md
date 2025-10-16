#  Enhancing Lightweight Image Super-Resolution through Frequency-Enhanced Mamba Networks


[Boyun Li](https://liboyun.github.io/), [Haiyu Zhao](https://pandint.github.io/), [Wenxin Wang](https://hi-wenxin.github.io/), [Peng Hu](https://penghu-cs.github.io/), [Yuanbiao Gou](https://ybgou.github.io/)\*, [Xi Peng](https://pengxi.me/)\*

> **Abstract:**  Recent advancements in Mamba have shown promising results in image restoration. These methods typically flatten 2D images into multiple distinct 1D sequences along rows and columns, process each sequence independently using selective scan operation, and recombine them to form the outputs. However, such a paradigm overlooks two vital aspects: i) the local relationships and spatial continuity inherent in natural images, and ii) the discrepancies among sequences unfolded through totally different ways. To overcome the drawbacks, we explore two problems in Mamba-based restoration methods: i) how to design a scanning strategy preserving both locality and continuity while facilitating restoration, and ii) how to aggregate the distinct sequences unfolded in totally different ways. To address these problems, we propose a novel Mamba-based Image Restoration model (MaIR), which consists of Nested S-shaped Scanning strategy (NSS) and Sequence Shuffle Attention block (SSA). Specifically, NSS preserves locality and continuity of the input images through the stripe-based scanning region and the S-shaped scanning path, respectively. SSA aggregates sequences through calculating attention weights within the corresponding channels of different sequences. Thanks to NSS and SSA, MaIR surpasses 40 baselines across 14 challenging datasets, achieving state-of-the-art performance on the tasks of image super-resolution, denoising, deblurring and dehazing.

## Environment & Dependencies

To ensure seamless execution of our project, the following dependencies are required:

* Python == 3.8.11
* Pytorch == 2.0.1
* cudatoolkit == 11.0.221

We export our conda virtual environment as environment.yaml. You can use the following command to create the environment.

```bash
conda env create -f environment.yaml
```

This ensures all dependencies are correctly installed, allowing you to focus on running and experimenting with the code.

## Datasets

The datasets used in our training and testing are orgnized as follows:

| Task                           | Training Set                                                                                                                                                                                                                                                                                                                                                                                                                                    |                                                                              Testing Set                                                                              | Visual Results |
|:-------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------: |
| lightweight Image SR           | LightSR:[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)<br />                                                                                                                                                                                                                                                                                                                                                                                |               Set5 + Set14 + BSD100 + Urban100 + Manga109 [[download](https://drive.google.com/file/d/1n-7pmwjP0isZBK7w3tx2y8CTastlABx1/view?usp=sharing)]               |  Coming Soon  |
| Gaussian Color Image Denoising | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) +  [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) + [BSD400](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) + [WED](http://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar) <br />[complete DFWB_RGB [download](https://drive.google.com/file/d/1jPgG_URDQZ4kyXaMMXJ8AZ8jEErCdKuM/view?usp=share_link)] |CBSD68 + Kodak24 + McMaster + Urban100  [[download](https://drive.google.com/file/d/1baLpOjNlTCNbREUDAZf9Lso6YCeUOQER/view?usp=sharing)]                 |  Coming Soon  | | [GoPro](https://drive.google.com/file/d/1abXSfeRGrzj2mQ2n2vIBHtObU6vXvr7C/view) + [HIDE](https://drive.google.com/file/d/1XRomKYJF1H92g1EuD06pCQe4o6HlwB7A/view?usp=sharing) |  [Download](https://drive.google.com/drive/folders/1cA3PgLYGTW_ofC8wPBR3DUlhpvuRDwuw?usp=sharing)  |
| Image Dehazing                 | Indoor & Outdoor:[RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-standard?authuser=0) (including 13990 indoor and 313950 outdoor images)<br />Mix: RESIDE-6K  (6000 images) <br />[Training Set [download](https://drive.google.com/drive/folders/1oaQSpdYHxEv-nMOB7yCLKfw2NDCJVtrx)]                                                                                                                                      |                                 SOTS / RESIDE-Mix [[download](https://drive.google.com/drive/folders/1oaQSpdYHxEv-nMOB7yCLKfw2NDCJVtrx)]                                 |  Coming Soon  |

### Training and Testing Commands on Super-Resolution

```bash
# Training commands for tiny version of x2, x3, x4 lightSR (<900K) (~3 days)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1268 basicsr/trainF.py -opt options/train/train_MaIR_T_lightSR_x2.yml --launcher pytorch
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1268 basicsr/trainF.py -opt options/train/train_MaIR_T_lightSR_x3.yml --launcher pytorch
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1268 basicsr/trainF.py -opt options/train/train_MaIR_T_lightSR_x4.yml --launcher pytorch

# Testing commands for small version of x2, x3, x4 lightSR (~1.3M)
python basicsr/test.py -opt options/test/test_MaIR_S_lightSR_x2.yml
python basicsr/test.py -opt options/test/test_MaIR_S_lightSR_x3.yml
python basicsr/test.py -opt options/test/test_MaIR_S_lightSR_x4.yml
```

### Training and Testing Commands on Color Image Denoising

```bash
# Training commands for Color Denoising with sigma=50
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=1268 basicsr/trainF.py -opt options/train/train_MaIR_CDN_s50.yml --launcher pytorch

# Testing commands for Color Denoising with sigma=50
python basicsr/test.py -opt options/test/test_MaIR_CDN_s50.yml

```

### Training and Testing Commands on Image Dehazing

```bash
# Training commands for Image Dehazing (~3 days)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=8 --master_port=1268 realDenoising/basicsr/trainF.py -opt realDenoising/options/train/train_MaIR_ITS.yml --launcher pytorch

# Testing commands for Image Dehazing
python realDenoising/basicsr/test.py -opt realDenoising/options/test/test_MaIR_ITS.yml

```

Cautions: torchrun is only available for pytorch>=1.9.0. If you do not want to use torchrun for training, you can replace it with `python -m torch.distributed.launch` for training.


## Acknowledgement

This code and README is based on [MambaIR](https://github.com/csguoh/MambaIR/) and [MaIR](https://github.com/XLearning-SCU/2025-CVPR-MaIR). Many thanks for their awesome work.
