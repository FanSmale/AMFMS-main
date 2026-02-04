<img width="3569" height="2401" alt="Noise_metrics" src="https://github.com/user-attachments/assets/27d6e2f6-3366-4e10-8978-2822ce86db3f" /># Enhancing Seismic Inversion Fidelity via Adaptive Multi-Frequency and Multi-Scale Fusion

## Abstract

Deep learning full waveform inversion (DL-FWI) has gained increasing attention due to its strong nonlinear fitting capability, reduced dependence on initial velocity models, and high inversion efficiency.
However, it often suffers from low fidelity due to weak spatial correlation, high complexity of geological structures, and poor training stability.
To address these issues, we propose an adaptive multi-frequency and multi-scale (AMFMS) algorithm.
For weak spatial correlation, we introduce the double convolution enhanced block with wavelet transform (DCEW) in the encoder, and the multi-frequency hierarchical adaptive refinement module (MFAR) in the decoder.
DCEW decomposes feature maps into multi-frequency components to enhance local detail preservation, while MFAR integrates multi-scale features through multi-frequency attention mechanisms to restore spatial correspondence.
Regarding complex geological structures, we design a new joint loss called MMT that combines $\mathcal{L}_1$, $\mathcal{L}_2$, and total variation regularization.
It enables a smooth transition from coarse-to-fine learning, balancing global structure consistency with local geological realism.
To enhance training stability, we adopt a dynamic learning rate schedule that integrates warm-up and cosine annealing strategies.
The warm-up phase ensures stable initialization, while the cosine annealing mechanism enhances convergence and model robustness.
Experiments are performed on three benchmark datasets from SEG salt, OpenFWI and Marmousi Ⅱ slice.
The results demonstrate that AMFMS outperforms six state-of-the-art methods across six evaluation metrics, achieving superior accuracy and structural fidelity, especially in complex and deep regions.


[Architecture_of_Net_new.pdf](https://github.com/user-attachments/files/25066549/Architecture_of_Net_new.pdf)


## Requirements
- Python 3.8
- torch 1.8.1+cu111
- torchvision 0.9.1+cu111

## How to use
- "model_test" - testing related methods
- "model_train_AMFMS" - training for AMFMS and AMFMS_SEG
- "model_train_TU-Net" - training for TU-Net and TU-Net-SEG
- "model_train_ABA" - training for ABA-FWI and ABA-FWI+
- "model_train_DDNet" - training for DD-Net and DD-Net70
- "model_train_InversionNet" - training for InversionNet
- "model_train_FCNVMB" - training for FCNVMB
- "param_config" - experimental parameter settings
- "path_config" - experimental path settings
