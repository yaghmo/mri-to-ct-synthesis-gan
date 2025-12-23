# MRI-to-CT Synthesis using GANs (Pix2Pix + NGF)

This project focuses on generating **high-fidelity CT images from MRI scans** using a **Conditional GAN (Pix2Pix)** architecture enhanced with **structural regularization (NGF – Normalized Gradient Fields)**.

The goal is to improve **anatomical accuracy** of the generated CT images, which is critical for **clinical applications such as radiotherapy planning**, where precise structural information is required.

---

## Project Overview

MRI-only clinical workflows aim to reduce patient exposure to radiation by avoiding CT acquisitions.  
However, CT images are still required for tasks that rely on accurate tissue density and structural boundaries.

This project addresses this challenge by:
- Learning a mapping from MRI to CT using deep learning
- Enforcing structural consistency between modalities
- Evaluating the impact of structural regularization on image quality

---

## Methodology

- **Model:** Conditional GAN (Pix2Pix)
  - Generator: U-Net (implemented with MONAI)
  - Discriminator: Multi-scale PatchGAN
- **Loss Functions:**
  - L1 reconstruction loss
  - Adversarial loss
  - Normalized Gradient Field (NGF) loss for structural consistency
- **Training Strategy:**
  - Comparison between models trained with and without NGF
  - High-fidelity variant using cropped training, anatomical masking, and background constraints
- **Inference:**
  - 2D slice-based training with 3D volumetric reconstruction at test time

---

## Data Processing

- Paired MRI–CT scans
- DICOM to NIfTI conversion
- Slice extraction and intensity normalization
- Orientation consistency checks
- Optional cranial masking for high-fidelity experiments

---

## Evaluation Metrics

Model performance is evaluated using:
- **SSIM (Structural Similarity Index Measure)**
  - Global SSIM
  - Local SSIM (within cranial region)
- **Dice coefficient** for structural overlap

---

## Key Results

- NGF regularization improved **global SSIM from 0.7134 to 0.7315**
- Structural overlap (Dice) increased from **0.8826 to 0.8851**
- High-fidelity variant achieved the **best local SSIM (0.6931)**

These results show that enforcing structural constraints leads to **more anatomically accurate CT synthesis**, suitable for clinical use.

---

## Technologies Used

- Python
- PyTorch
- MONAI
- Pix2Pix (GAN)
- Medical image processing (MRI / CT)

---

## Use Cases

- MRI-only clinical workflows
- Radiotherapy planning support
- Medical image synthesis research
- Structural image-to-image translation

---

## Disclaimer

This project is intended for **research and educational purposes only** and is **not a certified medical device**.

---

## Author

**Yaghmoracen Belmir**  
AI Engineer / Computer Vision  
