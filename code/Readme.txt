env.yml lists the dependencies. To be used with anaconda

ifcnn.py and swinir.py contain their corresponding models
ifcnn has been significantly changed though to allow for easier data handling and more concise code

model.py contains a Pytorch Module designed to stitch the fusion and denoising models together

Experiments.ipynb was/is used for experimenting with different parts of the setup
For now it contains a fully functional workflow for loading images and running the model

the snapshots folder should contain pretrained weights for ifcnn and swinir
as the files are too large for github, download them and place them in this folder
https://github.com/uzeful/IFCNN/blob/master/Code/snapshots/IFCNN-MAX.pth
https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth

the lytro folder contains on sample with two images from the ifcnn repository

the integrals folder contains one sample with 8 images generated with different focal heights