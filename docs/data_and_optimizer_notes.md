# Data preparation and optimizer notes (Luckshan)

## Data

- **Dataset:** GTSRB (German Traffic Sign Recognition Benchmark), 43 classes of
  road signs. I merged the official training set (26,640 images from
  torchvision) and the official test set (12,630 images), giving 39,270 images.
- **Resizing:** every image was resized to 64x64 RGB to imitate a low-resource
  camera sensor, as the assignment requires (maximum 64x64).
- **Split:** 70% train (27,489), 15% validation (5,890), 15% test (5,891).
  The split is stratified, so every class appears in all three sets in the same
  proportion. This matters because GTSRB is imbalanced: the largest classes have
  about 2,200 images and the smallest about 210.
- **Normalisation:** the per-channel mean and standard deviation come from the
  training set only, so no information from the validation or test sets leaks in.
- **Augmentation:** small random rotation (+/-10 degrees), shift (+/-10%) and zoom
  (+/-10%), applied to training images only. I did not use horizontal flips,
  because they would turn "turn left" signs into "turn right" signs.
- **Known limitation:** GTSRB training images come in sequences of about 30 frames
  of the same physical sign. A random split can put near-identical frames in both
  the training and test sets, so the very high accuracies (up to 99.8%) may be
  slightly optimistic.

## Optimizers (Model B, 20 epochs)

| Optimizer | Test accuracy |
|---|---|
| Adam, lr 0.001 | 98.64% |
| SGD, lr 0.01, momentum 0 | 42.30% |
| SGD, lr 0.01, momentum 0.5 | 66.66% |
| SGD, lr 0.01, momentum 0.9 | 98.13% |
| SGD, lr 0.01, momentum 0.99 | 99.13% |

- **Plain SGD** at lr 0.01 was far too slow: after 20 epochs it had only reached
  about 43% accuracy and the loss was still falling.
- **Momentum** keeps a running velocity, v = beta * v + gradient. When gradients
  point in the same direction the steps add up, so the effective step size is
  about lr / (1 - beta): 0.01, 0.02, 0.1 and 1.0 for beta = 0, 0.5, 0.9, 0.99.
  This is why larger momentum trained faster here. It also smooths noisy updates.
- **Adam** worked at its default lr 0.001 with no tuning and was the fastest in
  the first epochs. SGD with momentum 0.99 caught up around epoch 6-7 and finished
  about 0.5 points higher. That is about 29 test images, from a single run, so it
  may partly be noise.
- **Decision:** I kept Adam for the main experiments because it was chosen before
  the experiments and needs no tuning. I did not switch to SGD with momentum 0.99
  afterwards, to avoid choosing the optimizer on test-set results.
- **Caveat:** I did not tune the learning rate for either optimizer, so this
  compares sensible defaults, not the best each could do.