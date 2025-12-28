## U-NET - Image Segmentation Carvana

![OS](https://img.shields.io/badge/-Linux-grey?logo=linux) ![PyTorch](https://img.shields.io/badge/PyTorch-black?logo=PyTorch) ![License](https://img.shields.io/badge/License-MIT-green) ![Python](https://img.shields.io/badge/python-3.12%2B-blue)

---

End-to-end **image segmentation** project using a **U-Net architecture** implemented from scratch in **PyTorch**, trained on the **Carvana Image Masking Challenge** dataset (Kaggle).

The goal is to automatically segment the **car (foreground)** from the **background** in RGB images.


---

## Project Motivation

This project was developed as a **learning-oriented, hands-on Deep Learning project**, with the objective of **continuing to expand my knowledge of neural networks and computer vision** while working on a realistic, non-trivial task.

The main goals were:

- Deepen understanding of **convolutional neural networks (CNNs)**
- Implement an **encoder–decoder architecture (U-Net)** from scratch
- Work with **larger image segmentation datasets**
- Train models under **real-world hardware constraints**
- Build a complete pipeline: **training → inference → visualization**

The focus was not on achieving state-of-the-art performance, but on **clean implementation, reproducibility, and learning through practice**.

---


## Model Architecture: U-Net

A **classic U-Net architecture** is used, widely adopted for semantic segmentation tasks.

Key characteristics:
- Symmetric **encoder–decoder** structure
- **Skip connections** between encoder and decoder
- 2D convolutions with ReLU activations
- Binary segmentation output (foreground / background)

Implementation can be found in: `uNetUtils.py`

<p align = "center">
    <img src = "resources/UNET_architectureImage.png" alt = "MNIST dataset image" width = "400"/>
</p>


---

## Repo Structure

```bash
.
├── inference.py
├── trainuNet.py
├── uNetUtils.py
├── requirements.txt
├── resources/
│   └── inference_result.png
├── .gitignore
└── README.md

---

## Dataset 


This project uses the Carvana Image Masking Challenge dataset from Kaggle.

- RGB car images

- Binary segmentation masks

- Medium dataset (not included in the repository)

**Dataset** link:
https://www.kaggle.com/competitions/carvana-image-masking-challenge/data

```The dataset/ directory is excluded from the repository via .gitignore due to its size.```




