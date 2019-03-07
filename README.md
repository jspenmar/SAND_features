# Scale-Adaptive Neural Dense Features (SAND)

This repository contains the network architecture and pretrained feature 
models from "[Scale-Adaptive Neural Dense Features](https://www.researchgate.net/publication/331576664_Scale-Adaptive_Neural_Dense_Features_Learning_via_Hierarchical_Context_Aggregation)".
Additionally, we provide a simple script to load these models and run inference on a sample Kitti image.

## Citation
Please cite the following paper if you find SAND useful in your research:
```
@inproceedings{spencer2019,
  title={Scale-Adaptive Neural Dense Features: Learning via Hierarchical Context Aggregation},
  author={Spencer, Jaime  and Bowden, Richard and Hadfield, Simon},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

## Introduction

How do computers and intelligent agents view the world around them?
Feature extraction and representation constitutes one the basic building blocks towards answering this question. 
Traditionally, this has been done with carefully engineered hand-crafted techniques such as HOG, SIFT or ORB.
However, there is no "one size fits all" approach that satisfies all requirements.

In recent years, the rising popularity of deep learning has resulted in a myriad of end-to-end solutions to many computer vision problems. 
These approaches, while successful, tend to lack scalability and can't easily exploit information learned by other systems. 

Instead, we propose SAND features, a dedicated deep learning solution to feature extraction capable of providing hierarchical context information.
This is achieved by employing sparse relative labels indicating relationships of similarity/dissimilarity between image locations.
The nature of these labels results in an almost infinite set of dissimilar examples to choose from. 
We demonstrate how the selection of negative examples during training can be used to modify the feature space and vary it's properties.

To demonstrate the generality of this approach, we apply the proposed features to a multitude of tasks, each requiring different properties. 
This includes disparity estimation, semantic segmentation, self-localisation and SLAM.
In all cases, we show how incorporating SAND features results in better or comparable results to the baseline, whilst requiring little to no additional training.

<p align="center">
  <img src="images/sample_viz2.png">
</p>

## Prerequisites
- Python >= 3.6
- PyTorch >= 0.4 (not tested later versions)
- Imageio (from image reading)
- Matplotlib (for visualization)
- Sklearn (for PCA reduction)

## Usage
Breakdown for "[main.py](main.py)":
```
# Create and load pretrained SAND branch
model = Sand(32)
load_sand_ckpt(model, 'path/to/repo/models/32/ckpt_G.pt')
```

To visualize the features produced by the network:
```
features = model(img)
features = fmap2img(features_torch)  # Provided in "ops.py"
plt.imshow(features)
plt.show()
```

## Models
All models have been trained on a subset of the Kitti odometry sequence 00.
We provide **G**lobal, **L**ocal and hierarchical (**GL**) models for features of dimensionality 3, 10 & 32.
Models should be placed in "path/to/repo/models/" and can be loaded using torch.load()

3-D: &nbsp; [G](https://drive.google.com/file/d/1SGj2VHN78QaA5GWfOwxPbnj0M_-XgJwu/view?usp=sharing) --
[L](https://drive.google.com/open?id=1Mjhx21n0h78CoE6zrmREhRDMVHXOLX62) -- 
[GL](https://drive.google.com/open?id=194p6Kgw7KrrN1972CjNskPiOOkYwfQ7c)

10-D: [G](https://drive.google.com/open?id=1iu4l9L71VxdJtSIYPtVFxqjZeP-qTq1x) -- 
[L](https://drive.google.com/open?id=1tawKBU36-wHfrYShi-nV18HJH2mrexoL) -- 
[GL](https://drive.google.com/open?id=1ZUuHX8D9l2KlWEczUX0kr7VfcGAQ9wgD)

32-D: [G](https://drive.google.com/open?id=1TqsYNKR2jq3yW1TLXwXI2r8UBNn8Rqlr) -- 
[L](https://drive.google.com/open?id=1HuEwA70jht6XB3HYP5bz9mNXk4_RiJ4u) -- 
[GL](https://drive.google.com/open?id=12wYWOSqugSH9YGA4667SjF2hI5jqgV_C)

## Contact

You can contact me at [jaime.spencer@surrey.ac.uk](mailto:jaime.spencer@surrey.ac.uk)

