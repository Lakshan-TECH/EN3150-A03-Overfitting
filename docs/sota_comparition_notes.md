# SOTA Comparison Notes (Santhosh)

## 1. Fine-Tuning Setup

Two ImageNet-pretrained lightweight convolutional neural networks were selected for comparison with the custom CNN models:

- **MobileNetV2:** A lightweight CNN that uses inverted residual blocks and linear bottlenecks.
- **SqueezeNet 1.1:** A compact CNN that uses Fire modules to reduce the number of parameters.

Both models were loaded using the `torchvision` pretrained weights `IMAGENET1K_V1`. Their original 1000-class classification heads were replaced with new output layers for the 43 classes in the selected dataset.

### Model modifications

- **MobileNetV2:** The original classification layer was replaced with a linear layer mapping 1280 input features to 43 output classes.
- **SqueezeNet 1.1:** The final classification convolution was replaced with a 1 × 1 convolution mapping 512 channels to 43 output classes.

All layers were fine-tuned using Adam with a learning rate of 0.0005 for 20 epochs. The same training, validation, and test splits and augmentation strategy were used as for custom Models A and B.

Input images were resized or maintained at 64 × 64 pixels and normalised using the ImageNet channel means and standard deviations. The pretrained models were originally trained using 224 × 224 images, so using 64 × 64 inputs represents a resolution mismatch that may affect their performance.

The final evaluation used the held-out test set containing 5,891 images.

## 2. Test Results

The following results were obtained from the group's experiments.

| Metric | MobileNetV2 | SqueezeNet 1.1 | Model B | Model A |
|---|---:|---:|---:|---:|
| Parameters | 2,278,955 | 744,555 | 66,827 | 365,499 |
| FP32 model size | 8.92 MB | 2.86 MB | 0.28 MB | 1.40 MB |
| MACs per image | 24.50 M | 17.17 M | 4.16 M | 16.19 M |
| CPU latency per image | 6.86 ms | 3.54 ms | 1.99 ms | 1.94 ms |
| Training time per epoch | 108.9 s | 31.5 s | 23.1 s | 22.5 s |
| Test accuracy | 99.78% | 99.46% | 98.64% | 99.81% |
| Misclassified test images | 13 | 32 | 80 | 11 |

*Note: Model sizes, MAC counts, latency measurements, and training times depend on the measurement procedure, software environment, hardware, batch size, and model-saving format. Confirm the exact measurement definitions before including these figures in the final report.*

## 3. Comparison of the Two Pretrained Networks

MobileNetV2 achieved a test accuracy of 99.78%, while SqueezeNet 1.1 achieved 99.46%. MobileNetV2 therefore performed better in terms of classification accuracy in this experiment.

MobileNetV2 also converged rapidly because its pretrained weights already contained useful visual features. According to the recorded training results, it achieved 97.8% validation accuracy after the first epoch. SqueezeNet reached 96.2% validation accuracy after the third epoch.

However, MobileNetV2 has a larger computational and memory footprint.

- MobileNetV2 has approximately 3.06 times as many parameters as SqueezeNet 1.1.
- Its MAC count is approximately 1.43 times higher.
- Its measured training time per epoch is approximately 3.46 times higher.
- Its measured CPU inference latency is approximately 1.94 times higher.

SqueezeNet uses Fire modules, which combine squeeze layers and expand layers to reduce parameter requirements. Unlike MobileNetV2, its standard architecture does not primarily rely on depthwise separable convolutions.

These results illustrate that reducing parameter count does not necessarily reduce computation or inference latency by the same proportion. The type of operations used by an architecture and their implementation efficiency also affect runtime performance.

## 4. Comparison of Model B with the Pretrained Networks

Model B is the custom lightweight CNN built using depthwise separable convolutions. Its principal design objective is to achieve useful classification accuracy while maintaining a small model footprint and low computational cost.

### Model B versus MobileNetV2

- Model B has 66,827 parameters, compared with 2,278,955 for MobileNetV2.
- MobileNetV2 has approximately 34.1 times as many parameters.
- Model B has approximately 5.9 times fewer MACs per image.
- Its reported FP32 model file is approximately 32 times smaller.
- Model B achieved 98.64% test accuracy, compared with 99.78% for MobileNetV2.

MobileNetV2 achieved an accuracy advantage of 1.14 percentage points over Model B. In exchange, it required substantially more parameters, computation, memory, and training time.

### Model B versus SqueezeNet 1.1

- SqueezeNet has approximately 11.1 times as many parameters as Model B.
- Model B uses approximately 4.1 times fewer MACs per image.
- Its reported FP32 model file is approximately 10.2 times smaller.
- Model B achieved 98.64% test accuracy, compared with 99.46% for SqueezeNet.

SqueezeNet's accuracy advantage was 0.82 percentage points. Model B therefore provides a smaller and computationally cheaper alternative at the cost of a modest reduction in classification accuracy.

### Why MAC count does not directly determine latency

Model B's measured CPU inference latency was 1.99 ms per image. MobileNetV2 required 6.86 ms, while SqueezeNet required 3.54 ms.

Although depthwise separable convolutions reduce the number of multiply-accumulate operations, the speed improvement depends on hardware and software implementation. Depthwise operations may not be as efficiently parallelised or optimised as standard convolutions on some CPUs.

Therefore, MAC count should be treated as a computational-cost estimate rather than a direct measurement of execution time.

## 5. The Surprising Result: Model A versus MobileNetV2

Model A achieved 99.81% test accuracy, slightly higher than MobileNetV2's 99.78%, despite having far fewer parameters.

- Model A: 365,499 parameters and 99.81% test accuracy.
- MobileNetV2: 2,278,955 parameters and 99.78% test accuracy.

The accuracy difference is only 0.03 percentage points. On this dataset, the larger pretrained model did not demonstrate a clear accuracy advantage over the custom standard CNN.

One possible explanation is that the dataset is sufficiently learnable for a relatively small CNN. Model A may have enough representational capacity to extract the features required for this classification task.

However, this result does not establish that training from scratch is generally better than transfer learning. The small accuracy difference may reflect sampling variation, training randomness, preprocessing choices, or the particular dataset. Repeated runs and uncertainty estimates would be needed to establish whether the difference is meaningful.

## 6. Deployment Trade-Offs for Resource-Constrained Devices

Deployment on an embedded device requires consideration of model weights, intermediate feature maps, computation, latency, and available memory.

### Model storage

The reported FP32 file sizes are:

- Model B: 0.28 MB.
- SqueezeNet 1.1: 2.86 MB.
- MobileNetV2: 8.92 MB.

Quantisation can reduce storage requirements. For example, converting float32 weights to int8 weights reduces the storage per weight from 4 bytes to 1 byte, giving an approximate fourfold reduction in raw weight storage.

Based on the parameter counts, the raw int8 weight storage would be approximately:

- Model B: 65.3 KiB.
- SqueezeNet 1.1: 727.1 KiB.
- MobileNetV2: 2.18 MiB.

These are theoretical raw-weight estimates and exclude metadata, quantisation scales, biases where stored separately, and other file overhead. Actual deployment sizes must be measured after conversion.

A microcontroller with approximately 256–512 KB of RAM and 1–2 MB of flash could potentially accommodate a quantised version of Model B, depending on the memory required by its inference runtime and intermediate tensors. SqueezeNet may consume a substantial portion of the flash budget, while MobileNetV2 may exceed it in some configurations.

### Intermediate feature maps

Model weights are not the only source of memory usage. During inference, intermediate feature maps must also be stored.

For example, if Model B produces a feature map of size 16 × 64 × 64, it contains:

16 × 64 × 64 = 65,536 values.

This requires approximately 64 KiB when stored as int8 values or 256 KiB when stored as float32 values, before accounting for additional buffers.

Using a stride-2 initial convolution or another suitable downsampling operation could reduce the spatial dimensions and lower activation-memory requirements. However, the effect on classification accuracy must be evaluated.

### Recommended deployment scenarios

**For a microcontroller or a device with a strict memory budget:** Model B is a promising candidate because of its low parameter count and MAC count. Quantisation and activation-memory optimisation may further improve its suitability.

**For a Raspberry Pi or another device with more memory and processing capacity:** MobileNetV2 may be preferable when classification accuracy is the main priority and its runtime cost is acceptable.

The final deployment choice should be based on measurements on the intended target device rather than parameter count alone.

## 7. Limitations

Several limitations should be considered when interpreting these results.

1. **Single experimental run:** Each model was evaluated using one reported run. Small accuracy differences may be caused by random initialisation, data ordering, or other sources of experimental variation.

2. **Input resolution:** The pretrained models were evaluated at 64 × 64 pixels, although their original ImageNet training resolution was 224 × 224 pixels. This may affect the visual features available to the models.

3. **Dataset dependence:** High test accuracy on one dataset does not guarantee similar performance on other datasets or under real-world distribution shifts.

4. **CPU-specific timing:** Inference latency depends on CPU architecture, thread settings, batch size, framework version, and operator optimisation. Results may differ on GPUs or embedded hardware.

5. **Quantisation estimates:** The int8 model sizes discussed above are estimates based on raw parameter storage. Actual quantised models should be converted and measured using the intended deployment framework.

6. **Memory requirements:** Model file size alone does not determine whether a network fits on a microcontroller. Peak activation memory, runtime buffers, and operating-system or firmware overhead must also be considered.

7. **Resolution and preprocessing consistency:** The exact resizing method, image normalisation, channel conversion, and augmentation strategy should be documented to ensure a fair comparison.

8. **Accuracy uncertainty:** Differences of a few hundredths of a percentage point should not be treated as meaningful without repeated runs or statistical analysis.

## 8. Conclusion

The experimental results show that both pretrained lightweight architectures achieved high test accuracy, but they required more parameters and computational resources than the custom lightweight CNN.

Model B achieved 98.64% accuracy with 66,827 parameters, making it a strong candidate when model size and computation are important constraints. MobileNetV2 achieved 99.78% accuracy but required substantially more memory and computation. SqueezeNet 1.1 provided an intermediate trade-off.

The standard custom CNN, Model A, achieved the highest reported accuracy at 99.81%, demonstrating that a carefully designed model trained from scratch can perform competitively on a suitable dataset.

Overall, there is no universally best architecture. The appropriate choice depends on the required accuracy, memory budget, computational capacity, inference latency, and deployment environment. Model B is a promising choice for highly constrained devices, whereas MobileNetV2 may be more suitable when a small accuracy improvement justifies the additional resource requirements.

## 9. Reproducibility

The final report should include the following information to make the comparison reproducible:

- Dataset name, class count, and train/validation/test split.
- Model architecture and parameter count.
- Pretrained weight versions.
- Optimiser, learning rate, number of epochs, and batch size.
- Image resizing, normalisation, and augmentation procedures.
- Hardware and software environment.
- Model file format and size-measurement method.
- MAC-counting and CPU-latency measurement procedures.
- Test accuracy, confusion matrix, precision, and recall.
- Code and experiment results in the group's GitHub repository.
