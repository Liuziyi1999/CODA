# CODA
Consistency-Guided Multi-Source-Free Domain Adaptation  

### Overview
This repository is a PyTorch implementation of the paper.  

### Framework
![Framework](https://github.com/Liuziyi1999/CODA/blob/main/img/Framework.SVG)

### Prerequisites:
- python == 3.8
- pytorch == 1.11.0
- torchvision == 0.12.0
- numpy, scipy, sklearn, PIL, argparse

### Dataset
- Manually download the datasets [Office](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view), [Office-Caltech](http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar), [DomainNet](http://ai.bu.edu/M3SDA/) from the official websites.
- Move `gen_list.py` inside data directory.
- Generate '.txt' file for each dataset using `gen_list.py` (change dataset argument in the file accordingly).

### Training
- Train source models (shown here for Office with source A)

  ``` python train_source.py --dset office --s 0 --max_epoch 100 --trte val --gpu_id 0 --output ckps/source/ ```
- Adapt to target (shown here for Office with target D)

  ``` python adapt_multi.py --dset office --t 1 --max_epoch 15 --gpu_id 0 --output_src ckps/source/ --output ckps/adapt ``` 
