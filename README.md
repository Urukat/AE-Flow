# AE-Flow
This is an unofficial implementation of ICLR 2023 paper [AE-FLOW: Autoencoders with Normalizing Flows for Medical Images Anomaly Detection](https://openreview.net/forum?id=9OmCr1q54Z), which aims to replicate the paper. If you find nay issues about this repo, please contact [Pengfei Hu](feifei.hu@student.uva.nl) and Yikun Gu. Part of codes are from another repository [ae-flow](https://github.com/asiraudin/ae-flow).

## Installation
1. Install the conda to manage Python environment.
2. Install the specific libraries according to requirement file. 
```python
conda create -n ae-flow python=3.8
conda activate ae-flow
conda install pytorch==1.12.1
```
## Preparing datasets
1. [Chest X-Ray](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download)
Here we provided a processed version of Chest X-Ray: [link](https://drive.google.com/file/d/15jCc-zBHmB9ILcu6ACXEtdb1OetEiVjH/view?usp=sharing). While the processed data should be placed at **./src/data/chest_xray/processed**
## Getting started
```python
# firstly enter the root folder
cd src
# for training 
python train.py --submodel_name fast_flow --epochs 20 --model_path ./trained-model.pch --dataset_path ./data/chest_xray
# for testing
python test.py --submodel_name fast_flow --model_path ./trained-model.pch --dataset_path ./data/chest_xray
```
## Pretrained Model
| Subnets for flow architecture     |      Dataset      |  Link         |
|----------                         |:-----------------:|--------------:|
| Fast-flow                         |    Chest X-Ray    |     [link](https://drive.google.com/file/d/1DQgAklJeo_A6KRZoR0rL3uNn3LxoeXfG/view?usp=sharing)      |
| ResNet                            |    Chest X-Ray    |     [Unavailable]()      |

## TODO:
- [x] Draft for blogpost
- [x] Setup the runnable Environment
- [x] Download Chest X-ray dataset
- [x] Setup model architecture
- [x] Implement recon loss and flow loss
- [] Extend Different dataset (OCT/ISIS2018/ISIC2018/BraTS2021)
- [] Extend more evaluating metrics (AUC/F1/ACC/SEN/SPE)
- [] Utilize semi-supervised techniques to make use of abnormal data