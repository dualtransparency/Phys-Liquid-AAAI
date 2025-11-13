# Phys-Liquid: A Physics-Informed Dataset for Estimating 3D Geometry and Volume of Transparent Deformable Liquids

This repository contains the official implementation and sample data for our AAAI-26 paper:

> **Phys-Liquid: A Physics-Informed Dataset for Estimating 3D Geometry and Volume of Transparent Deformable Liquids**  
> Ke Ma et al., AAAI-26 (Oral)

## 1. Overview

- Physics-informed simulation dataset of transparent deformable liquids in lab scenes
- Four-stage reconstruction pipeline:
  1. Transparent-liquid segmentation
  2. Multi-view mask diffusion
  3. 3D reconstruction
  4. Mesh scaling to metric dimensions
- Benchmarks against InstantMesh and TriPoSR, plus evaluation on real-world DTLD data.

## 2. Dataset

- ~N images across multiple containers, colors, lighting conditions, scenes, and rotation modes.
- Each image is paired with:
  - RGB image
  - Liquid mask
  - 3D mesh (.obj) of the liquid
  - Full physical and rendering metadata
- This repo ships a **small sample** in `data/samples/`.
- The **full dataset** can be downloaded via:
  - (TBD) Zenodo / institutional server link
  - or `scripts/download_data.sh`

## 3. Installation

```bash
conda create -n phys-liquid python=3.10
conda activate phys-liquid
pip install -r requirements.txt
