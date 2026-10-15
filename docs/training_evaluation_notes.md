# Training and evaluation notes (Sampavi)

## Setup

- Both models: 20 epochs, batch size 128, Adam with learning rate 0.001,
  cross-entropy loss, the same data split and augmentation, seed 42.
- After every epoch the validation loss and accuracy were recorded. The weights
  with the best validation accuracy were restored before testing, so the test set
  was used only once per model.
- Everything was trained on a CPU, so all timings are CPU timings.

## Results on the test set (5,891 images)

| Metric | Model A | Model B |
|---|---|---|
| Accuracy | 99.81% | 98.64% |
| Misclassified images | 11 | 80 |
| Precision (macro) | 0.9981 | 0.9861 |
| Recall (macro) | 0.9969 | 0.9848 |
| F1 (macro) | 0.9975 | 0.9853 |
| Training time per epoch | 22.5 s | 23.1 s |
| CPU latency per image | 1.94 ms | 1.99 ms |

- Macro averages treat all 43 classes equally, so they are stricter than accuracy on
  an imbalanced dataset.

## Loss curves

- Neither model overfits. Validation loss stays below training loss because
  augmentation and dropout are active only during training.
- Model A passes 90% validation accuracy at epoch 2 and has essentially converged
  by epoch 10, with isolated dips such as epoch 14 (98.2%).
- Model B needs 7 epochs to pass 90%, and its validation loss is still falling at
  epoch 20 (0.049). Part of the 1.2-point gap to Model A may therefore be
  under-training and not only lower capacity. We did not test this with more epochs.

## Why Model B is not faster than Model A

- Model B has 3.9x fewer MACs, but on a CPU it took 23.1 s per epoch against 22.5 s,
  and 1.99 ms per image against 1.94 ms.
- Depthwise convolutions move about as much data as they compute on, so they are
  memory-bound, and PyTorch's CPU kernels for them are less optimised.
- In training, the augmentation and data handling cost the same for both models.
- The MAC saving should show up on hardware with efficient depthwise kernels, but we
  did not measure that.

## Confusion matrices

- Model A made only 11 errors, and its most frequent confusion happens just once, so
  there is no systematic weakness.
- Model B made 80 errors, mostly between speed-limit signs: 50 -> 30 (6 times),
  50 -> 80 (5), 70 -> 30 (3), 80 -> 60 (3), 60 -> 80 (3), 120 -> 100 (3).
  At 64x64 the digits differ by only a few pixels.
- Other Model B confusions: General caution -> Traffic signals (4) and
  Slippery road -> Wild animals crossing (3), which are all warning triangles.

## Lowest recall for Model B

| Class | Recall | Errors |
|---|---|---|
| Speed limit 20 | 93.5% | 2 of 31 |
| Slippery road | 94.8% | 4 of 77 |
| Bicycles crossing | 95.0% | 2 of 40 |
| Speed limit 50 | 96.5% | 12 of 338 |

- The first three are rare classes, so one error moves their recall by 1.3 to 3.2
  points. Speed limit 50 is the more reliable weak spot because it has many test images.
