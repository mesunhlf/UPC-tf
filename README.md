# [CVPR 2020] UPC: Learning Universal Physical Camouflage Attacks on Object Detectors

<p align="center"><img width="800"  src="/images/huang2019upc.gif"></p>

### Overview

This is the official Tensorflow implementation of the universal physical camouflage (UPC) method proposed in [Learning Universal Physical Camouflage Attacks on Object Detectors](https://arxiv.org/abs/1909.04326v1). The project page (including demo & dataset) is [here](https://mesunhlf.github.io/index_physical.html).

### Prerequisites

python **2.7**  
scipy **1.2.2**  
opencv-python **4.1.2**  
tensorflow **1.12rc1**  
easydict **1.6**
  
**Faster-RCNN** our code use the repository of [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn) for attacking. Please install this object detection  and make sure `tools/demo.py` can be run. We modify the `lib/nets/network.py` for feeding the tensors in our attacks and other files in `lib` need be copied in root directory.


### Run the Code
For changing the parameters, please refer to the settings in `flags.py`.   
For runing the training code, please execute `bash run_main.sh`.

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
	  title={UPC: Learning Universal Physical Camouflage Attacks on Object Detectors},
	  author={Lifeng Huang and Chengying Gao and Yuyin Zhou and Changqing Zou and Cihang Xie and Alan L. Yuille and Ning Liu},
	  booktitle={CVPR},
      year={2020}
	}
