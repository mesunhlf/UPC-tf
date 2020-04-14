# UPC-tf
### Overview

The code is repository for the universal physical camouflage (UPC) method described in [Learning Universal Physical Camouflage Attacks on Object Detectors](https://arxiv.org/abs/1909.04326v1).

Demo, dataset and more examples are demonstrated in [project page](https://mesunhlf.github.io/index_physical.html).

### Prerequisites

python **2.7**  
scipy **1.2.2**  
opencv-python **4.1.2**  
tensorflow **1.12rc1**  
easydict **1.6**
  
**faster-rcnn** our code use the repository of [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn) for attacking. Please install the object detection and make sure `tools/demo.py` can be run. We modify the `lib/nets/network.py` for feeding the tensors in our attacks and other files in `lib` need be copied in root directory.


### Run the Code
For changing the parameters, please look into `flags.py`.   
For runing the training code `bash run_main.sh`.

### Pipeline and Examples
<img src="/images/examples.jpg" width = "700" height = "300" align=center/>

### Dataset
We build the first standardized dataset, named [AttackScenes](https://drive.google.com/open?id=1tmzQj7Dm4zO4ROThDjJM5pJDrHMR2dWn), for evaluating the performace of physical attacks.  
**Environments** AttackScenes includes different virtual scenes
under various physical conditions.  
**Cameras** For each virtual scene, 18 cameras are
placed for capturing images from different viewpoints.  
**Illuminations** The illumination varies from dark to bright at 3 levels by controlling the strength of light sources.

### Citation
If you find this project useful, please consider citing:

	@article{Huang2019UPCLU,
	  title={UPC: Learning Universal Physical Camouflage Attacks on Object Detectors},
	  author={Lifeng Huang and Chengying Gao and Yuyin Zhou and Changqing Zou and Cihang Xie and Alan L. Yuille and Ning Liu},
	  journal={ArXiv},
	  year={2019},
	  volume={abs/1909.04326}
	}
