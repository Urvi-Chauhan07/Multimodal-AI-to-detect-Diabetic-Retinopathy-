# Multimodal-AI-to-detect-Diabetic-Retinopathy-

Deep learning models that combine **fundus photos** and **OCT scans** of the same patient to classify diabetic retinopathy (No DR / NPDR / PDR), with GAN-based augmentation and explainability (Grad-CAM, LIME, SHAP, Occlusion Sensitivity).

## Highlights
- Dual-encoder CNNs (EfficientNet-B0 / DenseNet-121) fuse fundus + OCT features
- CGAN/DCGAN used to synthesize minority-class images and balance training data
- Explainability shows per-modality contribution to each prediction

## Results (validation accuracy)
| Model | Accuracy |
|---|---|
| EfficientNet-B0 (dual) | 79.0% |
| DenseNet-121 (dual) | 84.0% |
| DenseNet-121 + GAN augmentation | **88.9%** |

## Requirements
`torch`, `torchvision`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `opencv-python`, `Pillow`, `shap`, `lime`, `scikit-image`

## Dataset
570 fundus+OCT image pairs, 217 patients, patient-level stratified split
