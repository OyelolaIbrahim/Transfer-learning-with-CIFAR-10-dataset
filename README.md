# Transfer Learning with CIFAR-10 — Feature Reuse Across Tasks

Demonstrating transfer learning by training a CNN on the first 
5 CIFAR-10 classes, freezing the learned feature layers, and 
reusing them to classify the remaining 5 classes — achieving 
comparable accuracy in significantly less training time.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green)
![Method](https://img.shields.io/badge/Method-Transfer%20Learning-purple)

---

## Overview

Implements transfer learning from scratch using CIFAR-10 
by splitting the dataset into two classification tasks — 
classes 0–4 (airplane, automobile, bird, cat, deer) and 
classes 5–9 (dog, frog, horse, ship, truck). A CNN 
(Model 1) is first trained on the first 5 classes. Its 
convolutional feature extraction layers are then frozen 
and reused in a second model (Model 2) trained only on 
the last 5 classes — demonstrating that learned visual 
features transfer effectively across related tasks while 
dramatically reducing training time.

---

## Problem Statement

Training deep neural networks from scratch is 
computationally expensive and requires large amounts of 
labelled data. Transfer learning solves this by reusing 
feature representations learned on one task to accelerate 
learning on a related task. This project demonstrates this 
principle concretely: the convolutional layers trained to 
recognise animals and vehicles in classes 0–4 transfer 
effectively to recognise the remaining 5 categories — 
without retraining those layers.

---

## Dataset

- **Name:** CIFAR-10 Image Classification Dataset
- **Source:** Built into Keras — loaded automatically via
  `keras.datasets.cifar10.load_data()` — no download needed
- **Total Size:** 60,000 colour images
  - Training: 50,000 images
  - Test: 10,000 images
- **Image Size:** 32×32 pixels, RGB (3 colour channels)
- **All 10 Classes:**

| Label | Class | Used In |
|-------|-------|---------|
| 0 | Airplane | Model 1 (source task) |
| 1 | Automobile | Model 1 (source task) |
| 2 | Bird | Model 1 (source task) |
| 3 | Cat | Model 1 (source task) |
| 4 | Deer | Model 1 (source task) |
| 5 | Dog | Model 2 (target task) |
| 6 | Frog | Model 2 (target task) |
| 7 | Horse | Model 2 (target task) |
| 8 | Ship | Model 2 (target task) |
| 9 | Truck | Model 2 (target task) |

> **No dataset download required.** CIFAR-10 is 
> downloaded automatically by Keras on first run.

---

## Approach

### Data Preparation
- Loaded CIFAR-10 using `cifar10.load_data()`
- Reshaped label arrays from `(50000, 1)` to 
  `(50000,)` using `.reshape()`
- Split dataset into two subsets:
  - **lt5:** Images with class labels 0–4 
    (source task — Model 1)
  - **gte5:** Images with class labels 5–9 
    (target task — Model 2)
- Re-labelled gte5 labels by subtracting 5 
  (so labels become 0–4 again for 5-class output)
- Visualised 20 random sample images from each 
  subset in a 2×10 grid

### Architecture Design
The model is split into two explicit groups of layers:

**Feature Layers — Convolutional backbone 
(shared between both models):**

| Layer | Type | Details |
|-------|------|---------|
| 1 | Conv2D | 64 filters, 3×3 kernel, valid padding, ReLU |
| 2 | Conv2D | 64 filters, 3×3 kernel, ReLU |
| 3 | MaxPooling2D | 2×2 pool size |
| 4 | Dropout | Rate = 0.25 |
| 5 | Flatten | Converts feature maps to 1D |

**Classification Layers — Task-specific dense head:**

| Layer | Type | Details |
|-------|------|---------|
| 6 | Dense | 128 units, ReLU |
| 7 | Dropout | Rate = 0.25 |
| 8 | Dense (output) | 5 units, Softmax |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimiser | Adam |
| Learning Rate | 0.001 |
| Loss Function | Categorical Cross-Entropy |
| Batch Size | 128 |
| Epochs | 20 |
| Normalisation | Pixel values divided by 255 |

### Transfer Learning Pipeline

**Step 1 — Train Model 1 on classes 0–4:**
```python
model_1 = Sequential(feature_layers + classification_layers)
train_model(model_1, (x_train_lt5, y_train_lt5), ...)
```

**Step 2 — Freeze all feature (convolutional) layers:**
```python
for layer in feature_layers:
    layer.trainable = False
```
Freezing sets `trainable = False` on all Conv2D 
and pooling layers — their weights are locked and 
will not be updated during Model 2 training.

**Step 3 — Build Model 2 with frozen features 
and retrain only the classification head:**
```python
model_2 = Sequential(feature_layers + classification_layers)
train_model(model_2, (x_train_gte5, y_train_gte5), ...)
```
Only the Dense classification layers are updated — 
the Conv2D feature extractors remain fixed.

