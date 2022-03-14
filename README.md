
<div align="center">
 
# Domain Generalization for Prostate Segmentation in Transrectal Ultrasound Images: A Multi-center Study

[![Paper](https://img.shields.io/badge/arXiv-2011.11390-brightgreen)]()
[![journal](https://img.shields.io/badge/Journal-Medical%20Image%20Analysis-red)]()

</div>

---
## Abstract
Prostate biopsy and image-guided treatment procedures are often performed under the guidance of ultrasound fused with magnetic resonance images (MRI). Accurate image fusion relies on accurate segmentation of the prostate on ultrasound images. Yet, the reduced signal-to-noise ratio and artifacts (e.g., spackle and shadowing) in ultrasound images limit the performance of automated prostate segmentation techniques and generalizing these methods to new image domains is inherently difficult. In this study, we address these challenges by introducing a novel 2.5D deep neural network for prostate segmentation on ultrasound images. Our approach addresses the limitations of transfer learning and finetuning methods (i.e., drop in performance on the original training data when the model weights are updated) by combining a supervised domain adaptation technique and a knowledge distillation loss. The knowledge distillation loss allows the preservation of previously learned knowledge and reduces the performance drop after model finetuning on new datasets. Furthermore, our approach relies on an attention module that considers model feature positioning information to improve the segmentation accuracy. We trained our model on 764 subjects from one institution and finetuned our model using only ten subjects from subsequent institutions. We analyzed the performance of our method on three large datasets encompassing  2067 subjects from three different institutions.
Our method achieved an average Dice Similarity Coefficient (Dice) of 94.0(+/-0.03) and Hausdorff Distance (HD95) of 2.28 mm in an independent set of subjects from the first institution. Moreover, our model generalized well in the studies from the other two institutions (Dice: 91.0(+/-0.03); HD95: 3.7 mm and Dice: 82.0(+/-0.03); HD95: 7.1 mm). We introduced an approach that successfully segmented the prostate on ultrasound images in a multi-center study, suggesting its clinical potential to facilitate the accurate fusion of ultrasound and MRI images to drive biopsy and image-guided treatments.

## Dataset
Our study included patients data from three independent cohorts acquired at Stanford Medicine, [UCLA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=68550661) and UCL, and it can not be shared at the moment. A detailed description of the training data can be found in the paper. For further information, please contact Dr.Mirabela Rusu ([Mirabela@stanford.edu](Mirabela@stanford.edu)).


## Dependencies
- Python 3.7
- PyTorch 1.7
- SimpleITK 1.2.2
- scikit-image
- scikit-learn

## Installation
 Please clone the repo as follows:
 
 ```
 git clone https://github.com/PIMED/TRUSGlandSegmentation/
 cd TRUSGlandSegmentation
```
To run the code without any OS compatibility issue the `environment.yml` is already exported (Windows 10). You can create the same environment as follows:
 ```
conda env create -f environment.yml
conda activate torch_gpu
```


## Train SPCNET On Your Data
Please place your MRI data and their corresponding labels in the dataset directory. To train SPCNET on your data, you can run train.py file as following:
 
 ```python -u train.py```
