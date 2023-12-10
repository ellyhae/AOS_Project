env.yml lists the dependencies. To be used with anaconda

ifcnn.py and swinir.py contain their corresponding models
ifcnn has been significantly changed though to allow for easier data handling and more concise code

model.py contains a Pytorch Module designed to stitch the fusion and denoising models together

Experiments.ipynb was/is used for experimenting with different parts of the setup
For now it contains a fully functional workflow for loading images and running the model

the snapshots folder contains pretrained weights for ifcnn and swinir

the lytro folder contains on sample with two images from the ifcnn repository

the integrals folder contains one sample with 8 images generated with different focal heights