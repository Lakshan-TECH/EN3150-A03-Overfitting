# Data preparation and optimizer notes (Luckshan)

## Data
- Dataset: GTSRB, 43 traffic-sign classes, 39,270 images used.
- Resized to 64x64 to simulate a low-resource sensor.
- Stratified 70/15/15 split so every class appears in all three sets.
- Normalisation uses training-set statistics only. No flips in augmentation,
  because left/right signs would change meaning.

## Optimizers (Model B)
- Adam (lr 0.001): 98.64% test accuracy.
- Plain SGD (lr 0.01): only 42.3% after 20 epochs, learning rate too small.
- SGD + momentum 0.9: 98.13%. SGD + momentum 0.99: 99.13%.
- Momentum multiplies the effective step by 1/(1-beta), which is why larger
  beta helped here.
