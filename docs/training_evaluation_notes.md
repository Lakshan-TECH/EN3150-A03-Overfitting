# Training and evaluation notes (Sampavi, Section 4)

This note summarises how Model A (standard CNN) and Model B (depthwise-separable CNN)
were trained and evaluated, what the curves and metrics show, and how to read them.

## 1. Training setup

| Setting | Value |
|---|---|
| Epochs | 20 for both models |
| Optimizer | Adam, learning rate 0.001, batch size 128 |
| Loss | Cross-entropy |
| Data | GTSRB, 64x64, stratified split 27,489 train / 5,890 validation / 5,891 test |
| Augmentation | Small rotation, shift and zoom, training images only |
| Checkpoint | Weights from the best validation-accuracy epoch were used for testing |
| Hardware | CPU only, so all timings are CPU timings |

The test set was touched once per model, after training, so it did not influence any choice.

## 2. How training progressed

| Epoch | Model A val. accuracy | Model B val. accuracy |
|---|---|---|
| 1 | 69.5% | 43.6% |
| 2 | 94.8% | 59.2% |
| 5 | 98.8% | 85.4% |
| 7 | 99.3% | 90.5% |
| 10 | 99.6% | 95.5% |
| 20 | 99.7% | 98.5% |

- Validation loss fell from 0.91 to 0.016 for Model A and from 1.78 to 0.049 for Model B.
- Model A had essentially converged by epoch 10. It showed one isolated dip at epoch 14
  (98.2%) that recovered immediately.
- Model B learned more slowly and was still improving at epoch 20, so its result is
  probably a little below what it could reach with more epochs. This was not tested.

**Overfitting check.** Neither model overfits. Validation accuracy is above training
accuracy at the end (Model A 99.7% against 98.7%, Model B 98.5% against 97.0%). This
is expected here, because dropout and augmentation are active only during training and
make the training score look worse than it is.

## 3. Test-set results

| Metric | Model A | Model B |
|---|---|---|
| Accuracy | 99.81% | 98.64% |
| Misclassified images (of 5,891) | 11 | 80 |
| Precision (macro) | 0.9981 | 0.9861 |
| Recall (macro) | 0.9969 | 0.9848 |
| F1 (macro) | 0.9975 | 0.9853 |

Macro averages weight all 43 classes equally, so rare classes count as much as common
ones. Both models have macro recall close to their accuracy, which means the errors are
not concentrated in a few classes.

The gap between the models is 1.17 percentage points, or 69 extra errors for Model B.
That is the price of having 5.5 times fewer parameters (66,827 against 365,499).

## 4. Speed

| | Model A | Model B |
|---|---|---|
| Training time per epoch | 22.5 s | 23.1 s |
| Single-image CPU latency | 1.94 ms | 1.99 ms |

Model B needs 3.9 times fewer multiply-accumulates, yet it was not faster. Depthwise
convolutions do little arithmetic per byte of memory they touch, so on a CPU they are
limited by memory movement, and the library's CPU code for them is less optimised than
for ordinary convolutions. In training, augmentation and data handling cost the same for
both models. The saving in MACs should matter more on hardware designed for it, but that
was not measured.

## 5. Errors

**Model A (11 errors).** Every confusion happened only once, for example Ahead only
predicted as Turn left ahead, and Speed limit 70 predicted as Speed limit 100. There is no
systematic weakness.

**Model B (80 errors).** The most frequent confusions are:

| True class | Predicted as | Times |
|---|---|---|
| Speed limit 50 | Speed limit 30 | 6 |
| Speed limit 50 | Speed limit 80 | 5 |
| General caution | Traffic signals | 4 |
| Slippery road, Speed limit 70, Speed limit 80, Speed limit 120, Speed limit 60 | Wild animals crossing, 30, 60, 100, 80 | 3 each |

Speed-limit signs dominate. At 64x64 pixels the digits differ by only a few pixels, and
the smaller model has less capacity to tell them apart. General caution, Traffic signals
and Slippery road are all warning triangles whose inner pictograms are small and similar.

**Weakest classes for Model B (recall).**

| Class | Recall | Errors |
|---|---|---|
| Speed limit 20 | 93.5% | 2 of 31 |
| Slippery road | 94.8% | 4 of 77 |
| Bicycles crossing | 95.0% | 2 of 40 |
| Speed limit 50 | 96.5% | 12 of 338 |

The first three classes have few test images, so a single error moves their recall by
1.3 to 3.2 points. Speed limit 50 is the more reliable weak spot, because it has many test
images and is the main source of confusions.

## 6. Limitations

- Each model was trained once (seed 42), so differences of a few tenths of a percent could
  be noise.
- Near-identical frames of the same physical sign may appear in both training and test
  sets, so the accuracies may be slightly optimistic.
- Timings are from a CPU and are only a proxy for a microcontroller.

## 7. Ideas that were not tested

- Train Model B for more epochs, since its loss was still falling.
- Use a learning-rate schedule, or SGD with momentum 0.99, which did well in the optimizer study.
- Use a larger input size for the speed-limit classes, where the digits are hard to read.

## 8. Questions this note should let me answer

1. **Why was the best-validation checkpoint used?** To pick the strongest epoch without
   looking at the test set.
2. **Why is validation accuracy above training accuracy?** Dropout and augmentation act
   only during training.
3. **Why is Model B not faster despite fewer MACs?** Its operations are memory-bound on a
   CPU, so fewer multiplications do not translate directly into speed.
4. **Why does Model B confuse speed limits?** The digits differ by a few pixels at 64x64,
   and it has less capacity than Model A.
5. **Why use macro-averaged precision and recall?** They treat every class equally, which
   is stricter than accuracy on an imbalanced dataset.
