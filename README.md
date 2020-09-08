# RUDA implemented in PyTorch
To appear at ECML-PKDD2020 https://ecmlpkdd2020.net/ 
(Best (student) ML paper award)

Based on implementation from https://github.com/thuml/CDAN

## Folders
- pytorch (where .py are stored) 
- data (where data is stored) 
	- 1 folder for each dataset.
	- .txt file with path to images are stored here.

## Prerequisites
- PyTorch >= 0.4.0 (with suitable CUDA and CuDNN version)
- torchvision >= 0.2.1
- Python3
- Numpy
- argparse
- PIL

## Data
- Digits : https://drive.google.com/file/d/1Y0wT_ElbDcnFxtu25MB74npURwwijEdT/view?usp=drive_open
- Office31 : https://people.eecs.berkeley.edu/~jhoffman/domainadapt/
- Office-Home : http://hemanthdv.org/OfficeHome-Dataset/

## Training
python train_image_submission.py

Details about arguments: python train_image_submission.py -h
