# EN3150 Assignment 03 - Resource-Constrained CNN for Edge Image Classification

**Group: Overfitting** | University of Moratuwa, EN3150

| Index no. | Name | GitHub |
|---|---|---|
| 230016K | ABISHEK L. | [AbishekLesley](https://github.com/AbishekLesley) |
| 230380T | LUCKSHAN S.M. | [Lakshan-TECH](https://github.com/Lakshan-TECH) |
| 230572J | SAMPAVI G. | [sampavany](https://github.com/sampavany) |
| 230581K | SANTHOSH S. | [Santhosh-04-S](https://github.com/Santhosh-04-S) |

## What this project does
Traffic-sign classification on **GTSRB** (43 classes, images resized to 64x64, stratified 70/15/15 split) with
two custom CNNs and two fine-tuned lightweight networks:

* **Model A** - standard CNN (365,499 parameters, 16.19 M MACs)
* **Model B** - depthwise-separable CNN (66,827 parameters, 4.16 M MACs, under the 100k limit)
* **MobileNetV2** and **SqueezeNet 1.1** - ImageNet pre-trained, fine-tuned on the same splits

## Main results (test set, 5,891 images)

| Model | Parameters | fp32 size | MACs | Test accuracy |
|---|---|---|---|---|
| Model A | 365,499 | 1.40 MB | 16.19 M | 99.81% |
| Model B | 66,827 | 0.28 MB | 4.16 M | 98.64% |
| MobileNetV2 | 2,278,955 | 8.92 MB | 24.50 M | 99.78% |
| SqueezeNet 1.1 | 744,555 | 2.86 MB | 17.17 M | 99.46% |

Full tables: [results/results_tables.md](results/results_tables.md). Figures: [results/figures](results/figures).
## Key findings

- **Accuracy vs size.** Model A (365,499 parameters) reached 99.81% test accuracy and
  Model B (66,827 parameters) reached 98.64%, which is 11 against 80 errors on 5,891
  test images.
- **A small from-scratch CNN matched a pretrained network.** Model A scored 99.81%,
  against 99.78% for fine-tuned MobileNetV2, which has about 6x more parameters (2.28 M).
- **Model B is much smaller.** It has 34x fewer parameters than MobileNetV2 and about
  65 KB of weights in int8 (estimated), at a cost of 1.14 accuracy points.
- **Fewer MACs did not mean faster on a CPU.** Model B needs 3.9x fewer MACs than
  Model A, but single-image CPU latency was 1.99 ms against 1.94 ms.
- **Optimizers.** Plain SGD (lr 0.01) reached only 42.3% in 20 epochs. Momentum 0.9
  reached 98.1% and momentum 0.99 reached 99.1%, slightly above Adam (98.6%).
- **Where Model B fails.** Mostly between speed-limit signs (for example 50 vs 30 and
  80), because the digits differ by only a few pixels at 64x64.
- **Limitations.** One run per model, CPU-only timings, int8 sizes estimated, and
  near-identical frames of the same sign may appear in both the training and test
  sets, so the accuracies may be slightly optimistic.

## Documentation

- Full report: [docs/Overfitting_A03_EN3150.pdf](docs/Overfitting_A03_EN3150.pdf)
- Data and optimizer notes: [docs/data_and_optimizer_notes.md](docs/data_and_optimizer_notes.md)
- Architecture notes: [docs/architecture_notes.md](docs/architecture_notes.md)
- Training and evaluation notes: [docs/training_evaluation_notes.md](docs/training_evaluation_notes.md)
- SOTA comparison notes: [docs/sota_comparison_notes.md](docs/sota_comparison_notes.md)
  
## How to run
```bash
pip install -r requirements.txt
python gtsrb_edge_cnn.py                                   # full run (downloads GTSRB, ~1.5 h on CPU)
python gtsrb_edge_cnn.py --epochs 2 --fake-data --skip-sota   # quick smoke test, no download
```
Outputs (figures, confusion matrices, per-class CSVs, results.json) are written to `./outputs` (not tracked by git).

## Repository layout
```
gtsrb_edge_cnn.py   training / evaluation script (all tasks)
requirements.txt
results/            result tables, per-class metrics, figures used in the report
docs/               report PDF and notes
```
