File overvie:

- env.yml:	lists the dependencies. To be used with anaconda

- ifcnn.py:
- swinir.py:
- swin2sr.py:	contain their corresponding models
		ifcnn has been significantly changed though to allow for easier data handling and more concise code

- model.py:	contains a Pytorch Module designed to stitch the fusion and denoising models together

- Experiments.ipynb:	was/is used for experimenting with and testing different parts of the setup
			For now it contains a fully functional workflow for loading images and running the model

- AOS_integrator_dataset.ipynb:	an adaptation of a notebook from the AOS repository
				can be used to convert a folder of areal images to a folder/dataset of integral focal stacks
				must be placed in the AOS\AOS for Drone Swarms\LFR\python folder of the AOS project and run with the corresponding anaconda environment

- integrals:	contains a small dataset genereated by AOS_integrator_dataset.ipynb
		[id]_integrals.tiff 	is a collection of integrals at different focal lengths. All integrals for one sample are stored in a single file, hopefully speeding up load times
		[id]_gt.png 		is a copy of the corresponding ground thruth image
		missing.txt 		lists all samples for which too few files were present

- dataset.py:	a pytroch dataset class to load generated integral stacks and prepare them for the model

- snapshots:	this folder has been used so far to store pretrained weights for ifcnn and swinir
		their size prevents the gihub upload, but as these pretrained weights are not useful for our usecase, it's likely not necessary to download the yourself
		if you'd still like to, they can be found here:
		https://github.com/uzeful/IFCNN/blob/master/Code/snapshots/IFCNN-MAX.pth
		https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth

- lytro:	contains a sample with two images from the ifcnn repository
		only used for initial ideas presentation to have data similar to the training dataset for pretrained weights
		can likely be removed soon


Model Considerations:

IFCNN as described in it's paper uses pretrained resnet weights for it's first layer.
This forces them to use 3 channels for grayscale images, just repeating them along the axis.
It probably also forces us to do specific input handling, e.g normalization, which need to be looked into (TODO)

IFCNN's output also has 3 channels, which are converted to grayscale using PIL if needed.
SwinIR can be customized in how many channels it uses and what value range the image has ([0,1] or [0,255]).
For the first demonstration using pretrained weights, a derivable version of PIL's grayscale conversion was implemented and used.
The resulting "channel flow" was therefore: 3 > IFCNN > 3 > conversion > 1 > SwinIR > 1
However, there may be better ways to do this. Here are some ideas:
- Because pretrained resnet weights may not be useful here, we could go completely grayscale, with 1 channel everywhere and no need for conversion (could make training harder)
- replace the imitated PIL conversion with a per pixel Conv layer (i.e. (1,1) kernel) or adjust the final IFCNN layer for the same effect
- use 3 channels throughout, removing the need for conversion in the middle, only converting at the end (would make training harder due to more weights)

General Notes:

Much of this is simply bodged together to get the model working for the initial presentation.
Therefore, please look into and adjust the code and parameters as we move away from these pretrained weights.
Don't just trust the initial implementation to have done everything correctly!