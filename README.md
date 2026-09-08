# DSCPNet: Dynamic Scale Fusion and Cascaded Prototype Refinement Network

This repository provides a PyTorch implementation of DSCPNet for few-shot sports image classification. DSCPNet is designed to address three common challenges in sports images: scale variation, background distraction, and fine-grained inter-class ambiguity under limited labeled samples.

## Overview

DSCPNet contains three main modules:

- **Dynamic Scale Fusion Attention (DSFA)**, which fuses multi-scale contextual features with spatial and channel selection.
- **Cascaded Foreground-Scene Interaction (CFSI)**, which models interactions between foreground-aware and scene-aware feature responses.
- **Cross-Prototype Feature Refinement (CPFR)**, which refines support prototypes and query representations for metric-based few-shot classification.

The default backbone is ResNet12. The model is trained on base classes and evaluated on novel classes under an N-way K-shot episodic protocol.

## Requirements

The code was prepared with the following environment:

```bash
python==3.8.13
torch==1.10.0
torchvision
numpy
pillow
```

Install the required packages with:

```bash
pip install torch torchvision numpy pillow
```

Please install the PyTorch version that matches your CUDA environment.

## Dataset Structure

Organize the dataset as follows:

```text
data/
  base/
    class_1/
      image_001.jpg
      image_002.jpg
    class_2/
      image_001.jpg
      image_002.jpg
  novel/
    class_1/
      image_001.jpg
      image_002.jpg
    class_2/
      image_001.jpg
      image_002.jpg
```

Only images from `base/` are used during meta-training. Images from `novel/` are reserved for meta-testing and are not used to update model parameters or tune hyperparameters.

## File Description

```text
config.py   Default configuration.
data.py     Image-folder dataset loader and episodic sampler.
model.py    ResNet12 backbone and DSCPNet modules, including DSFA, CFSI, and CPFR.
engine.py   Training, evaluation, accuracy, confidence interval, and seed utilities.
train.py    Training entry point.
test.py     Evaluation entry point.
```

## Training

Run training with:

```bash
python train.py --data_root ./data --shot 1 --query 15 --epochs 50 --checkpoint ./dscpnet_resnet12.pth
```

For 5-shot training, use:

```bash
python train.py --data_root ./data --shot 5 --query 15 --epochs 50 --checkpoint ./dscpnet_resnet12_5shot.pth
```

## Evaluation

Run evaluation with 600 randomly sampled test episodes:

```bash
python test.py --data_root ./data --shot 1 --query 15 --test_episodes 600 --checkpoint ./dscpnet_resnet12.pth
```

For 5-shot evaluation, use:

```bash
python test.py --data_root ./data --shot 5 --query 15 --test_episodes 600 --checkpoint ./dscpnet_resnet12_5shot.pth
```

The evaluation script reports the mean accuracy, standard deviation, and 95% confidence interval:

```text
95% CI = 1.96 * std / sqrt(number of test episodes)
```

## Notes

- The implementation follows the main structure described in the manuscript and is intended for reproducible few-shot sports image classification experiments.
- The default setting uses 5-way K-shot episodes with 15 query images per class.
- The code assumes that each selected class contains at least `shot + query` images.
- For a fair comparison with other methods, the same backbone, data splits, image size, and episodic evaluation protocol should be used.

## Code Availability

The source code and implementation details supporting the study are provided in this repository.
