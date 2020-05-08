# [CVPR 2020] Universal Physical Camouflage Attacks on Object Detectors

<p align="center"><img width="800"  src="/images/huang2019upc.gif"></p>

### Overview

This is the official Tensorflow implementation of the universal physical camouflage (UPC) method proposed in [Universal Physical Camouflage Attacks on Object Detectors](https://arxiv.org/abs/1909.04326v1). The project page (including demo & dataset) is [here](https://mesunhlf.github.io/index_physical.html).

### Prerequisites

python **2.7**  
scipy **1.2.2**  
opencv-python **4.1.2**  
tensorflow **1.12rc1**  
easydict **1.6**
  
**Faster-RCNN** our code uses the repository of [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn) as an example to attack.   
(1) please download the repository from [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn) and compile the code according to its `README.md`.  
(2) run the `tools/demo.py` to test the installation.  
(3) copy the folder `lib` from `tf-faster-rcnn` into the root directory of `UPC-tf` except `lib/nets/network.py` file, which is uploaded and modified for feeding the tensor in our pipeline.


### Run the Code
Download the training data from this [link](https://drive.google.com/open?id=1gP-DXTByjteK6o5o8PfgosKevF3Mh4T4).  
For changing the parameters, please refer to the settings in `flags.py`.   
For runing the training code, please execute `bash run_main.sh`.


### Test the Patterns
Requirements: [Autodesk 3dsMax 2018](https://www.autodesk.com/products/3ds-max/overview) and [V-ray 3.6](https://www.chaosgroup.com/vray/3ds-max).  
(1) train the camouflage patterns  
(2) download the dataset [AttackScenes](https://drive.google.com/open?id=1tmzQj7Dm4zO4ROThDjJM5pJDrHMR2dWn) and release the `human.zip` and `scenes.zip` to get  human models and scenes files.  
(3) use Autodesk 3dsMax software to paste (i.e. uv mapping function) the trained patterns on human models.  
(4) export the human model as (.fbx) format from 3dsMax software and import it into each virtual scene (.3max).  
(5) install the V-ray 3.6 plug-ins (for 3dsMax), and render the images from pre-defined cameras.  
(6) run the faster-rcnn demo `tools/demo.py` to test the rendered images.

We also provide the rendered images, please click [here](https://drive.google.com/open?id=1BLiQfRs49HkfEWRX3SEeLexGIqso3Q_E). 


### Pipeline and Examples
<img src="/images/examples.jpg" align=center/>


### Dataset
We collect the first standardized dataset, named [AttackScenes](https://drive.google.com/open?id=1tmzQj7Dm4zO4ROThDjJM5pJDrHMR2dWn), for fairly evaluating the performace of physical attacks under a controllable and reproducible environment.

**Environments** AttackScenes includes different virtual scenes under various physical conditions.  
**Cameras** For each virtual scene, 18 cameras are placed for capturing images from different viewpoints.  
**Illuminations** The illuminations are accessible at 3 levels, and can be adjusted by controlling the strength of light sources.


### Citation
If you find this project is useful for your research, please consider citing:

	@inproceedings{Huang2020UPC,
	  title={Universal Physical Camouflage Attacks on Object Detectors},
	  author={Lifeng Huang and Chengying Gao and Yuyin Zhou and Cihang Xie and Alan L. Yuille and Changqing Zou and Ning Liu},
	  booktitle={CVPR},
      year={2020}
	}
