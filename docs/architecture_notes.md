\# Architecture notes (Abishek)



\## Model A: standard CNN (365,499 trainable parameters, 16.19 M MACs)



\- Four blocks of \[Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2], with channels

&#x20; 3 -> 16 -> 32 -> 64 -> 128, then Flatten -> FC 128 -> FC 43.

\- Counting rules: a conv layer has k\*k\*Cin\*Cout weights (no bias, because BatchNorm

&#x20; follows), BatchNorm has 2\*C parameters, a fully connected layer has in\*out + out.

\- Example: Conv3 = 3\*3\*32\*64 = 18,432 weights + 128 BatchNorm = 18,560.

\- FC1 takes 128\*4\*4 = 2,048 inputs, so it has 2,048\*128 + 128 = 262,272 parameters.

\- FC1 alone is 72% of the whole model, so the flatten-then-FC head is the main

&#x20; memory cost, not the convolutions.



\## Model B: depthwise-separable CNN (66,827 trainable parameters, 4.16 M MACs)



\- A standard 3x3 stem convolution (3 -> 16), then four depthwise-separable blocks

&#x20; (16->32, 32->64, 64->128, 128->256). Each block is a depthwise 3x3 convolution

&#x20; followed by a pointwise 1x1 convolution.

\- Depthwise: k\*k\*Cin parameters. Pointwise: Cin\*Cout parameters.

\- Global average pooling replaces Flatten, then FC 64 -> FC 43. This removes the

&#x20; 262k-parameter FC1, so part of the parameter saving comes from the pooling and

&#x20; not only from the separable convolutions.

\- The convolutional part of Model B has 47,584 parameters against 97,680 in Model A.

\- Total: 66,827, which is under the 100,000 limit.



\## Why MACs fall from 16.19 M to 4.16 M



\- A standard 3x3 conv costs 9\*Cin\*Cout MACs per output position. The separable

&#x20; version costs 9\*Cin + Cin\*Cout, a ratio of 1/Cout + 1/9. For Cout = 256 that is

&#x20; about 0.115, roughly 8.7x fewer MACs in that layer.

\- The whole-network saving is smaller (3.9x), because the standard stem convolution

&#x20; at 64x64 resolution (1.77 M MACs) is still 43% of Model B's compute, and the 1x1

&#x20; pointwise layers still cost the most.

\- MACs do not equal speed: on our CPU Model B was not faster than Model A.



\## Activation functions



\- ReLU is max(0, x) and ReLU6 is min(max(0, x), 6). Each is a single comparison,

&#x20; so they need no multiplier and no exponential, and run in one cycle on a

&#x20; microcontroller.

\- Sigmoid, tanh, Swish and GELU all need exp(), which is expensive on hardware

&#x20; without a floating-point unit, so they were avoided.

\- ReLU6 limits activations to \[0, 6]. That small fixed range makes 8-bit

&#x20; quantisation accurate, which is why Model B uses it.

\- Softmax appears only inside the training loss. At inference the prediction is

&#x20; the argmax of the logits, so no exponential is computed on the device.



\## Limitations



\- Possible train/test leakage: GTSRB has about 30 frames of the same physical sign,

&#x20; and a random split can put near-identical frames in train and test. The very high

&#x20; accuracies (up to 99.8%) may be slightly optimistic.

\- Every model was trained once (seed 42), so differences of a few tenths of a

&#x20; percent could be noise.

\- All timings were measured on a CPU, so they are only a proxy for a microcontroller.

\- The int8 sizes are estimates (1 byte per parameter), not measured after real

&#x20; quantisation.

\- Model B differs from Model A in two ways (separable convolutions and global

&#x20; average pooling), so we cannot say how much each one contributes.

