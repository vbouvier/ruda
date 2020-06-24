# RUDA implemneted in PyTorch

## Folders
- pytorch (where code is run) 
- data (where data is stored) 
	- 1 folder for each dataset.
	- txt file with path to images are stored here.

## Prerequisites
- PyTorch >= 0.4.0 (with suitable CUDA and CuDNN version)
- torchvision >= 0.2.1
- Python3
- Numpy
- argparse
- PIL

## Data
Digits : https://drive.google.com/file/d/1Y0wT_ElbDcnFxtu25MB74npURwwijEdT/view?usp=drive_open
Office31 : https://people.eecs.berkeley.edu/~jhoffman/domainadapt/
Office-Home : http://hemanthdv.org/OfficeHome-Dataset/

## Training
To start training: 
python train_image_submission.py

Details about arguments: python train_image_submission.py -h