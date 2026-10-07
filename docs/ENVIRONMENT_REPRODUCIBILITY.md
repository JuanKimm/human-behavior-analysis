# DD-R01 raw-video environment reproducibility

## Validated preprocessing profile: `pc_colab_preprocessing_v1`
Raw-video parity was empirically sensitive to the preprocessing stack. The following versions are therefore an exact contract for the current PC/Colab candidate:

- NumPy `2.2.6`
- OpenCV distribution `opencv-python==4.11.0.86` (imported `cv2.__version__ == 4.11.0`)
- Ultralytics `8.4.70`

Torch is platform-specific. Semantic parity was observed with Windows CPU `torch 2.13.0+cpu / torchvision 0.28.0+cpu` and Colab CUDA `torch 2.11.0+cu128 / torchvision 0.26.0+cu128` after the preprocessing trio above was aligned.

## Evidence
For the same Validation video `3AI_Office_01::Video (2)`, the unaligned Colab environment produced a different 2D detection mask and an 8-frame TCN transition shift. After aligning NumPy/OpenCV/Ultralytics, Local CPU and Colab CUDA had the same detected mask, the same pose-quality ratios, and identical TCN/ST-DS/DBN labels and transition decision frames. The residual numeric differences in 3D skeleton/probabilities were sub-micro scale and did not change semantic outputs.

## Scope
This profile is the current **PC/Colab raw-video** contract. It is not a Jetson/JetPack lock. Jetson Orin must receive its own validated environment profile during the later edge-integration stage.

## Rule
Official raw-video runs must pass the preprocessing environment contract. Engineering diagnostics may use the explicit override only when the deviation is documented and the resulting outputs are not treated as release evidence.
