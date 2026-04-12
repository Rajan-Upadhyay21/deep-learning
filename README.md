Deep Learning in PyTorch — From Fundamentals to Production-Ready Architectures

A structured, beginner-to-advanced deep learning reference repository implemented entirely in PyTorch — covering tensor computation, automatic differentiation, feedforward networks, convolutional architectures, recurrent models, transfer learning, and professional training pipelines, each demonstrated through clean, self-contained, executable programs on real benchmark datasets.


Overview
This repository is a comprehensive, ground-up implementation of the deep learning stack — built deliberately in PyTorch, the industry-standard framework for neural network research and production deployment. Every program is architected as a focused, self-contained learning module: one concept, one file, one clear demonstration.
The progression moves from first principles — tensor algebra and automatic differentiation — through the canonical neural network architectures that underpin modern AI: feedforward networks, convolutional networks for spatial data, recurrent and LSTM networks for sequential data, and pretrained ResNet fine-tuning via transfer learning. Each implementation is validated against a real benchmark dataset (MNIST, CIFAR-10, Iris, synthetic time-series), with documented accuracy benchmarks and training dynamics.
The repository serves three audiences simultaneously: students building intuition for the first time, practitioners who need clean reference implementations, and engineers constructing a portfolio that demonstrates end-to-end deep learning proficiency — from raw tensor operations to GPU-accelerated, checkpointed training pipelines.

Benchmark Results
#FileConceptDatasetResult1tensor_operations.pyTensor algebra, broadcasting, GPU——2autograd_demo.pyAutomatic differentiation, gradient computation——3simple_nn.pyFeedforward neural networkIris~97% accuracy4deep_nn.pyDeep NN + BatchNorm + DropoutMNIST~98% accuracy5cnn_image_classifier.pyConvolutional neural networkCIFAR-10~82% accuracy6transfer_learning.pyFine-tuning pretrained ResNet18CIFAR-10~90%+ accuracy7rnn_text.pyVanilla RNN for sequence modelingSynthetic sequences—8lstm_timeseries.pyLSTM time-series forecastingSine waveMSE < 0.0019training_loop.pyReusable pipeline + early stoppingMNIST~98% accuracy10save_load_model.pyModel checkpointing and resuming——

What This Repository Covers
Tensors & Automatic Differentiation

Tensor creation — torch.tensor, torch.zeros, torch.ones, torch.rand, torch.randn, torch.arange
Shape manipulation — reshape, view, squeeze, unsqueeze, permute, transpose
Mathematical operations — element-wise arithmetic, matrix multiplication (@, torch.matmul), broadcasting rules
Indexing and slicing — standard, advanced, and boolean mask indexing
GPU acceleration — .to(device), torch.cuda.is_available(), device-agnostic code patterns
Automatic differentiation with autograd — requires_grad, computational graph construction, .backward(), .grad
Gradient computation — chain rule through multi-operation graphs, .detach(), torch.no_grad()
Practical autograd — manual gradient verification and comparison with analytical derivatives

Feedforward Neural Networks

nn.Module architecture — __init__, forward, layer composition
nn.Linear — weight initialization, bias terms, in/out feature dimensions
Activation functions — ReLU, Sigmoid, Tanh, Softmax — behavioral differences and appropriate use cases
Loss functions — CrossEntropyLoss, MSELoss, BCELoss, BCEWithLogitsLoss — selection criteria
Optimizers — SGD (with momentum), Adam, AdamW, RMSProp — convergence characteristics
Multi-class classification on the Iris dataset — data loading, normalization, training, and evaluation

Deep Networks with Regularization

Stacking multiple hidden layers — depth vs. width tradeoffs
Batch Normalization (nn.BatchNorm1d, nn.BatchNorm2d) — internal covariate shift, training stability, placement conventions
Dropout (nn.Dropout) — stochastic neuron deactivation, model.train() vs. model.eval() behavior
Weight decay / L2 regularization — weight_decay parameter in optimizers
nn.Sequential — compact architecture definition for linear pipelines
Deep NN achieving ~98% accuracy on MNIST with full regularization stack

Convolutional Neural Networks

nn.Conv2d — kernel size, stride, padding, channel dimensions, receptive field intuition
nn.MaxPool2d — spatial downsampling, translation invariance
Feature map dimensions — tracking (batch, channels, H, W) through the full forward pass
nn.Flatten — transition from spatial feature maps to classification head
Data augmentation — RandomCrop, RandomHorizontalFlip, ColorJitter, Normalize with dataset statistics
End-to-end image classification pipeline on CIFAR-10 achieving ~82% accuracy
Visualizing learned convolutional filters and intermediate feature maps

Transfer Learning with Pretrained Models

Loading pretrained ResNet18 from torchvision.models
Freezing pretrained backbone weights — param.requires_grad = False
Replacing the classification head — model.fc = nn.Linear(512, num_classes)
Fine-tuning strategy — frozen backbone + trainable head vs. full network unfreezing
Achieving ~90%+ accuracy on CIFAR-10 with minimal additional training
Understanding why transfer learning dramatically outperforms training from scratch on limited data

Recurrent Networks & Sequential Modeling

Vanilla RNN (nn.RNN) — hidden state propagation, sequence-to-one prediction, vanishing gradient problem
Long Short-Term Memory (nn.LSTM) — cell state, hidden state, input/forget/output gate mechanics
batch_first=True — input tensor shape conventions (batch, seq_len, input_size)
Sequence preprocessing — normalization, windowing for time-series, sliding window construction
LSTM time-series forecasting on synthetic sine wave data — MSE below 0.001
Understanding when to use RNN vs. LSTM vs. Transformer for sequential tasks

Production-Grade Training Infrastructure

Reusable, modular training loop — separated train_epoch and eval_epoch functions
Early stopping — validation loss patience tracking with configurable threshold
Learning rate schedulers — StepLR, CosineAnnealingLR, ReduceLROnPlateau — when to use each
Gradient clipping — torch.nn.utils.clip_grad_norm_ — preventing exploding gradients in RNNs
Model checkpointing — torch.save(model.state_dict(), path) and model.load_state_dict(torch.load(path))
Training resumption — saving and restoring full training state including optimizer and scheduler
Mixed precision training with torch.cuda.amp.autocast and GradScaler for GPU acceleration


Repository Structure
bashdeep-learning/
├── 01_basics/
│   ├── tensor_operations.py        # Tensor creation, shapes, broadcasting, math ops, GPU transfer
│   └── autograd_demo.py            # Computational graphs, .backward(), gradient accumulation, no_grad
│
├── 02_neural_networks/
│   ├── simple_nn.py                # Feedforward NN on Iris — data loading, training, evaluation
│   └── deep_nn.py                  # Deep NN with BatchNorm + Dropout on MNIST (~98% accuracy)
│
├── 03_convolutional_networks/
│   ├── cnn_image_classifier.py     # Custom CNN on CIFAR-10 with data augmentation (~82% accuracy)
│   └── transfer_learning.py        # ResNet18 fine-tuning — frozen backbone + custom head (~90%+)
│
├── 04_recurrent_networks/
│   ├── rnn_text.py                 # Vanilla RNN — sequence modeling, hidden state propagation
│   └── lstm_timeseries.py          # LSTM forecasting on sine wave — windowed input, MSE < 0.001
│
├── 05_training_utils/
│   ├── training_loop.py            # Reusable pipeline — early stopping, LR scheduling, metrics logging
│   └── save_load_model.py          # Checkpointing — state_dict save/load, full training state resumption
│
├── requirements.txt                # Pinned dependencies for reproducible environment setup
└── README.md

Getting Started
Prerequisites

Python 3.10 or higher
NVIDIA GPU with CUDA support recommended (CPU training supported for all programs)
pip or conda package manager

Installation
bash# Clone the repository
git clone https://github.com/Rajan-Upadhyay21/deep-learning.git
cd deep-learning

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU-only installation
pip install torch torchvision torchaudio

# Install remaining dependencies
pip install numpy matplotlib scikit-learn

# Or install everything at once
pip install -r requirements.txt
Running Any Program
bash# Fundamentals
python 01_basics/tensor_operations.py
python 01_basics/autograd_demo.py

# Neural Networks
python 02_neural_networks/simple_nn.py
python 02_neural_networks/deep_nn.py

# Convolutional Networks
python 03_convolutional_networks/cnn_image_classifier.py
python 03_convolutional_networks/transfer_learning.py

# Recurrent Networks
python 04_recurrent_networks/rnn_text.py
python 04_recurrent_networks/lstm_timeseries.py

# Training Utilities
python 05_training_utils/training_loop.py
python 05_training_utils/save_load_model.py

Sample Code
Device-Agnostic Training Setup
pythonimport torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Every tensor and model must live on the same device
model = MyNetwork().to(device)

for X_batch, y_batch in train_loader:
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)
    predictions = model(X_batch)

Autograd — Watching PyTorch Compute Gradients
pythonimport torch

# Scalar computation graph
x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(4.0, requires_grad=True)

z = x ** 2 + 2 * y         # z = x² + 2y

z.backward()                # dz/dx = 2x = 6.0
                            # dz/dy = 2   = 2.0

print(x.grad)               # tensor(6.)
print(y.grad)               # tensor(2.)

# This is exactly what happens across millions of parameters
# during neural network backpropagation — PyTorch handles it automatically

CNN Forward Pass — Tracking Tensor Shapes
pythonimport torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # (B, 3, 32, 32) → (B, 32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # → (B, 32, 16, 16)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → (B, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # → (B, 64, 8, 8)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                  # → (B, 4096)
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)                    # → (B, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = ConvNet().to(device)
x = torch.rand(8, 3, 32, 32).to(device)   # batch of 8 CIFAR-10 images
print(model(x).shape)                      # torch.Size([8, 10])

Transfer Learning — Minimal Code, Maximum Performance
pythonfrom torchvision import models
import torch.nn as nn

# Load ResNet18 pretrained on ImageNet (1.2M images, 1000 classes)
model = models.resnet18(pretrained=True)

# Freeze all backbone parameters — preserve learned ImageNet representations
for param in model.parameters():
    param.requires_grad = False

# Replace classification head — only this layer trains
model.fc = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 10)      # 10 CIFAR-10 classes
)

# Optimizer targets only the trainable head parameters
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Result: ~90%+ accuracy on CIFAR-10 vs ~82% training from scratch
# Training time: minutes instead of hours

Early Stopping — Preventing Overfitting Automatically
pythonclass EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, checkpoint_path='best_model.pth'):
        self.patience       = patience
        self.min_delta      = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_loss      = float('inf')
        self.counter        = 0

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            return False    # continue training
        else:
            self.counter += 1
            return self.counter >= self.patience   # True = stop training

Learning Path
Follow this sequence for the most effective progression through the material:
Stage 1 — Foundations (01_basics/)
  └── tensor_operations.py    ← master tensor algebra before anything else
  └── autograd_demo.py        ← understand gradients before touching nn.Module

Stage 2 — Neural Networks (02_neural_networks/)
  └── simple_nn.py            ← first full training loop on tabular data
  └── deep_nn.py              ← add BatchNorm + Dropout, scale to MNIST

Stage 3 — Convolutional Networks (03_convolutional_networks/)
  └── cnn_image_classifier.py ← spatial feature extraction on CIFAR-10
  └── transfer_learning.py    ← leverage pretrained representations

Stage 4 — Sequential Models (04_recurrent_networks/)
  └── rnn_text.py             ← vanilla RNN, understand the vanishing gradient
  └── lstm_timeseries.py      ← LSTM gating mechanism, time-series forecasting

Stage 5 — Production Infrastructure (05_training_utils/)
  └── training_loop.py        ← early stopping, LR scheduling, metrics tracking
  └── save_load_model.py      ← checkpointing for long training runs

                  ▼
    Ready for Transformers, Diffusion Models, and LLM Fine-Tuning

Key Concepts Reference
ConceptPrecise DefinitionTensorA generalized n-dimensional array — the fundamental data structure for all PyTorch computationAutogradPyTorch's automatic differentiation engine — tracks operations on tensors to compute exact gradients via the chain ruleEpochOne complete forward and backward pass over the entire training datasetBatch SizeNumber of training samples processed in a single forward/backward pass before a weight updateLearning RateThe scalar multiplier controlling the step size taken in the direction of the negative gradientOverfittingA model that has memorized training data idiosyncrasies and fails to generalize to unseen examplesDropoutA regularization technique that randomly zeros neuron activations during training, forcing redundant representationsBatch NormalizationNormalizes layer inputs to zero mean and unit variance per mini-batch — stabilizes training and permits higher learning ratesCNNA neural architecture that applies learned spatial filters to extract hierarchical visual features from image dataLSTMA recurrent architecture with gating mechanisms (input, forget, output) that selectively retain and discard sequential information across long time horizonsTransfer LearningInitializing a model with weights pretrained on a large dataset, then adapting it to a target task — dramatically reducing required training data and computeGradient ClippingCapping gradient magnitudes before the optimizer step — prevents exploding gradients in deep or recurrent networks

Requirements
txttorch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
numpy>=1.23.0
matplotlib>=3.6.0
scikit-learn>=1.2.0
bashpip install -r requirements.txt

Roadmap

 Transformer architecture — self-attention, multi-head attention, positional encoding from scratch
 Vision Transformer (ViT) — patch-based image classification without convolutions
 Generative Adversarial Networks (GANs) — DCGAN on CIFAR-10 with training stability techniques
 Variational Autoencoders (VAEs) — latent space learning and image generation
 Object detection — YOLO-style single-shot detection pipeline
 Weights & Biases integration — real-time experiment tracking, hyperparameter sweeps
 ONNX model export — framework-agnostic model serialization for cross-platform deployment
 TorchScript — JIT compilation for production-grade inference optimization


References

PyTorch Official Documentation
Deep Learning — Goodfellow, Bengio, Courville
CS231n: Convolutional Neural Networks for Visual Recognition — Stanford
fast.ai: Practical Deep Learning for Coders
The Illustrated Transformer — Jay Alammar


Author
Rajan M Upadhyay
MS Computer Science — Roosevelt University
LinkedIn · GitHub · rajanupadhyay2121@gmail.com
