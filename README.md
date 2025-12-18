# Optional Homework — Advanced Topics in Neural Networks

## Summary
This project trains a small model that takes CIFAR-10 RGB images (3×32×32) and outputs grayscale 28×28 images that are horizontally and vertically flipped.

What I implemented:
- **Dataset**: `CustomDataset` loads CIFAR-10 and generates targets using the reference pipeline: resize → grayscale → hflip → vflip.
- **Model**: `TransformNet` (very small) does: resize (bilinear) → 1×1 conv for grayscale → hflip + vflip → clamp to [0, 1].
- **Training**: MSE loss + AdamW optimizer, 90/10 train/val split, TensorBoard logging, early stopping.
- **Qualitative results**: Script exports 5 pairs of images (ground truth vs prediction) into `outputs/`.
- **Benchmarking**: `inference.py` benchmarks sequential transforms on CPU vs model inference on CPU/CUDA for multiple batch sizes and DataLoader settings.

## Expected Points
- **(1) Model architecture/design**: 3/3
- **(2) Loss function + motivation**: 2/2
- **(3) Early stopping + motivation**: 2/2
- **(4) ≥5 model outputs vs ground truth**: 1/1
- **(5) Benchmark on CPU and GPU/MPS**: 1/2 (?)

**Total expected**: **9/10**
