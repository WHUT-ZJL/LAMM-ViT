# LAMM-ViT: Official PyTorch Implementation

This repository contains the official PyTorch implementation for the paper:

**LAMM-ViT: AI Face Detection via Layer-Aware Modulation of Region-Guided Attention**  
*Jiangling Zhang (First Author)*  
Wuhan University of Technology  
Contact: `340056@whut.edu.cn`

**Paper accepted at ECAI 2025.**

[![arXiv](https://img.shields.io/badge/arXiv-2505.07734-b31b1b.svg)](https://arxiv.org/abs/2505.07734)

## Overview

LAMM-ViT is a novel Vision Transformer architecture designed for robust AI-generated face detection. It excels at generalizing across various generation techniques (like GANs and Diffusion Models) by focusing on structural inconsistencies between facial regions. The model integrates two key components:
1.  **Region-Guided Multi-Head Attention (RG-MHA):** Uses facial landmarks to guide attention towards specific facial areas.
2.  **Layer-aware Mask Modulation (LAMM):** Dynamically adjusts the regional focus at different network depths, capturing hierarchical forgery cues.

## Requirements

To run this code, you need Python 3.8+ and the following libraries. You can install them using pip:

```bash
pip install torch torchvision tqdm numpy opencv-python Pillow scikit-learn h5py dlib psutil
```

You will also need the dlib facial landmark predictor model. The `feature_extraction.py` script will attempt to download it automatically. If it fails, you can download it manually from [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), unzip it, and place `shape_predictor_68_face_landmarks.dat` in the root directory of this project.

## Usage Workflow

The workflow is divided into four main steps: Data Preparation, Preprocessing, Feature Extraction, Training, and Inference.

### 1. Data Preparation

Organize your real and AI-generated (fake) face datasets into the following directory structure:

```
/path/to/your/datasets/
├── real/
│   └── IMDB_WIKI/
│       ├── image001.jpg
│       └── ...
└── fake/
    ├── StyleGAN3/
    │   ├── image001.png
    │   └── ...
    ├── StableDiffusion1.5/
    │   ├── image002.jpg
    │   └── ...
    └── ... (other fake datasets)
```

### 2. Preprocessing (Optional but Recommended)

This script will remove invalid/small images and resize all valid images to a standard size (e.g., 224x224), overwriting the original files.

Run the script for each dataset directory you want to process:

```bash
python preprocess.py --directory /path/to/your/datasets/fake/StyleGAN3
```

### 3. Feature Extraction

This script processes the images to extract facial landmarks and region masks, saving them into HDF5 files for efficient loading during training.

Update the dataset paths inside the `feature_extraction.py` script and then run it:

```bash
python feature_extraction.py --data_root /path/to/your/datasets --output_dir ./features
```

This will create a `features/` directory with the same structure as your data, containing `.h5` files for `train`, `val`, and `test` splits.

### 4. Training

To train the LAMM-ViT model, configure the parameters inside the `train.py` script (e.g., `feature_dir`, `output_dir`, `datasets` to use) and run it. The script supports both single-GPU and multi-GPU (DDP) training.

```bash
# For single-GPU training
python train.py

# For multi-GPU training
torchrun --nproc_per_node=NUM_GPUS train.py
```

The best model weights will be saved as `best_model.pth` in the specified output directory.

### 5. Inference

To evaluate a trained model on various test sets, configure the paths in `inference.py` (model path, feature directory) and run it.

```bash
python inference.py
```

The script will print the Accuracy (ACC) and Average Precision (AP) for each test dataset and save a detailed `prediction_results.json` file.

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@misc{zhang2025lammvitaifacedetection,
      title={LAMM-ViT: AI Face Detection via Layer-Aware Modulation of Region-Guided Attention}, 
      author={Jiangling Zhang and Weijie Zhu and Jirui Huang and Yaxiong Chen},
      year={2025},
      eprint={2505.07734},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.07734}, 
}
```
