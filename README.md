# AAIProject: CycleGAN (Selfie2Anime)

This project is a modernized implementation of CycleGAN with Spectral Normalization and Class Activation Mapping (CAM) Attention, migrated to **PyTorch** for compatibility with modern hardware (Mac M-series and Google Colab T4).

## 🚀 Key Features
- **Framework:** PyTorch (replaces retired MXNet/GluonCV).
- **Hardware Acceleration:**
  - **Google Colab:** Automatic detection of NVIDIA T4 GPU (CUDA).
  - **Modern Macs:** Native support for Apple Silicon GPU (MPS).
- **Task:** Specialized for Selfie-to-Anime translation.

## 🛠 Setup & Installation

### 1. Requirements
Install the necessary dependencies:
```bash
pip install -r requirements.txt
```

### 2. Pre-trained Weights
The project uses weights originally from the `ufownl/cycle_gan` repository. 
If you have the MXNet `.params` file (e.g., in `model/selfie2anime.gen_ab.params`), you must convert it to PyTorch format.

**On Google Colab (recommended for conversion):**
```bash
pip install mxnet torch
python3 convert_weights.py
```
This will generate `model/selfie2anime.gen_ab.pth`.

## 📸 Usage

### Selfie to Anime CLI Demo
To transform your own photos, run:
```bash
python3 selfie2anime.py path/to/your/selfie.jpg
```
The script uses **dlib** for automatic face detection and alignment.

### Training
To train from scratch:
```bash
python3 train.py --dataset selfie2anime
```

## 📂 Project Structure
- `pix2pix_gan.py`: PyTorch model architectures (Generator/Discriminator).
- `selfie2anime.py`: Inference script with face detection.
- `dataset.py`: PyTorch DataLoaders and image utilities.
- `convert_weights.py`: Utility to port weights from MXNet.
- `requirements.txt`: Modern dependency list.

## 📝 References
- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://junyanz.github.io/CycleGAN/)
- [U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization](https://arxiv.org/abs/1907.10830)
- [Original MXNet Implementation](https://github.com/ufownl/cycle_gan)
