# SOTA comparison notes

## Fine-tuning setup

- Two ImageNet-pretrained networks: MobileNetV2 and SqueezeNet 1.1 (torchvision
  weights IMAGENET1K_V1).
- The 1000-class output layer was replaced by a new 43-class layer (Linear 1280 -> 43
  for MobileNetV2, a 1x1 convolution 512 -> 43 for SqueezeNet).
- All layers were fine-tuned with Adam at learning rate 0.0005 for 20 epochs, on the
  same train/validation/test split and augmentation as Models A and B.
- Inputs were normalised with the ImageNet mean and standard deviation, and kept at
  64x64. Both networks were pretrained on 224x224, so this is a scale mismatch.

## Results on the test set (5,891 images)

| Metric | MobileNetV2 | SqueezeNet 1.1 | Model B | Model A |
|---|---|---|---|---|
| Parameters | 2,278,955 | 744,555 | 66,827 | 365,499 |
| fp32 size | 8.92 MB | 2.86 MB | 0.28 MB | 1.40 MB |
| MACs per image | 24.50 M | 17.17 M | 4.16 M | 16.19 M |
| CPU latency | 6.86 ms | 3.54 ms | 1.99 ms | 1.94 ms |
| Training time per epoch | 108.9 s | 31.5 s | 23.1 s | 22.5 s |
| Test accuracy | 99.78% | 99.46% | 98.64% | 99.81% |
| Errors (of 5,891) | 13 | 32 | 80 | 11 |

## Comparing the two pretrained networks

- MobileNetV2 was more accurate, and it converged fastest: 97.8% validation accuracy
  after the first epoch, thanks to the pretrained features. SqueezeNet reached 96.2%
  after epoch 3.
- MobileNetV2 has 3.1x the parameters of SqueezeNet and 1.4x the MACs, and it took
  3.5x longer per epoch and 1.9x longer per image.
- SqueezeNet has Fire modules and no depthwise convolutions, so it spends more MACs
  per parameter.

## Model B against the pretrained networks

- Model B gives up 1.14 accuracy points to MobileNetV2 and 0.82 points to
  SqueezeNet 1.1.
- In return it has 34x fewer parameters than MobileNetV2 (11x fewer than
  SqueezeNet), a 32x smaller file, and 5.9x fewer MACs (4.1x fewer than SqueezeNet).
- Its measured CPU latency gap is smaller (3.4x against MobileNetV2) than its MAC
  gap, because PyTorch does not turn the MAC saving of depthwise layers into speed
  on a CPU.

## The surprising result

- Model A (365,499 parameters, trained from scratch) matched MobileNetV2 (2.28 M
  parameters) in accuracy: 99.81% against 99.78%. On this dataset, capacity beyond a
  few hundred thousand parameters bought almost nothing.

## Deployment trade-offs

- A typical microcontroller has about 256-512 KB of RAM and 1-2 MB of flash. Model B
  in int8 (about 65 KB) fits easily. SqueezeNet (about 0.7 MB) would take most of the
  flash, and MobileNetV2 (about 2.2 MB) does not fit without pruning or a smaller
  width.
- Weights are only half of the memory: the largest feature map also has to fit in
  RAM. In Model B the stem output is 16x64x64 = 65,536 values (64 KB in int8). A
  stride-2 stem would cut that by 4x.
- For a microcontroller we would deploy Model B. For a device with megabytes of
  memory (for example a Raspberry Pi) where accuracy matters most, fine-tuned
  MobileNetV2 is the safer choice and needs no architecture design.

## Limitations

- Single run per model, so small accuracy differences may be noise.
- The pretrained networks were used at 64x64 instead of the 224x224 they expect,
  which disadvantages them.
- Timings are on a CPU, and int8 sizes are estimates.
