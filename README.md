# PointDenoise-Net
This project is modified from Pointcleannet to denoise more complex models
## Prerequisites
* CUDA and CuDNN (changing the code to run on CPU should require few changes)
* Python 2.7
* PyTorch 1.0

## Setup
Install required python packages, if they are not already installed ([tensorboardX](https://github.com/lanpa/tensorboard-pytorch) is only required for training):
``` bash
pip install numpy
pip install scipy
pip install tensorboardX
```
Clone this repository:
``` bash
git clone https://github.com/schwien/PointDenoise-Net.git
cd pointcleannet
```

Download datasets:
``` bash
cd data
python download_data.py --task denoising

In the datasets the input and ground truth point clouds are stored in different files with the same name but with different extensions.
- For denoising: `.xyz` for input noisy point clouds, `.clean_xyz` for the ground truth.

## Denoising
To denoise point clouds using default settings:
``` bash
cd denoise
mkdir results
./run.sh
```
(the input shapes and number of iterations are specified in run.sh file)

## Training
To train PCPNet with the default settings:
``` bash
python train_pcpnet.py
```
## Citation
The article has not been published yet and will be supplemented later.

