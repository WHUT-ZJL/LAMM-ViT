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
