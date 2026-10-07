## Table 1 - Model A vs Model B (test set)

| Metric | Model A (standard) | Model B (depthwise-separable) |
|---|---|---|
| Trainable parameters | 365,499 | 66,827 |
| MACs / image (M) | 16.19 | 4.16 |
| Model size on disk, fp32 (KB) | 1437.4 | 282.4 |
| Estimated size, int8 (KB) | 356.9 | 65.3 |
| Training time / epoch (s, CPU) | 22.5 | 23.1 |
| CPU latency, 1 thread (ms) | 1.94 | 1.99 |
| Test accuracy | 0.9981 | 0.9864 |
| Precision (macro) | 0.9981 | 0.9861 |
| Recall (macro) | 0.9969 | 0.9848 |
| F1 (macro) | 0.9975 | 0.9853 |

## Table 2 - Optimizer comparison on Model B

| Optimizer | Best val acc | Final val acc | Min val loss | Epochs to 90% val acc | Test acc |
|---|---|---|---|---|---|
| Adam lr=0.001 | 0.9876 | 0.9849 | 0.0466 | 7 | 0.9864 |
| SGD lr=0.01 (m=0) | 0.4275 | 0.4275 | 1.7583 | not reached | 0.4230 |
| SGD lr=0.01 (m=0.5) | 0.6596 | 0.6596 | 0.9951 | not reached | 0.6666 |
| SGD lr=0.01 (m=0.9) | 0.9786 | 0.9786 | 0.0714 | 10 | 0.9813 |
| SGD lr=0.01 (m=0.99) | 0.9907 | 0.9907 | 0.0335 | 7 | 0.9913 |

## Table 3 - Custom Model B vs fine-tuned lightweight SOTA (test set)

| Metric | Model B | MobileNetV2 | SqueezeNet1.1 |
|---|---|---|---|
| Trainable parameters | 66,827 | 2,278,955 | 744,555 |
| Model size fp32 (MB) | 0.276 | 8.924 | 2.859 |
| Estimated size int8 (MB) | 0.064 | 2.173 | 0.710 |
| MACs / image (M) | 4.16 | 24.50 | 17.17 |
| CPU latency, 1 thread (ms) | 1.99 | 6.86 | 3.54 |
| Training time / epoch (s, CPU) | 23.1 | 108.9 | 31.5 |
| Test accuracy | 0.9864 | 0.9978 | 0.9946 |
| Precision (macro) | 0.9861 | 0.9981 | 0.9947 |
| Recall (macro) | 0.9848 | 0.9971 | 0.9941 |
| F1 (macro) | 0.9853 | 0.9976 | 0.9943 |
