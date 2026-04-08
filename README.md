# deep-learning
Deep Learning implementations 
📌 About This Repository
This repository is a structured, beginner-to-advanced Deep Learning study guide implemented entirely in PyTorch. Each program is self-contained, well-commented, and designed to teach one key concept at a time.
Whether you're a student, researcher, or developer, this repo helps you:

🔬 Understand the theory behind each model
💻 See clean, runnable code for every concept
📈 Train models on real datasets (MNIST, CIFAR-10, Iris, etc.)
🏗️ Build a strong foundation for advanced deep learning
deep-learning/
│
├── 📂 01_basics/
│   ├── tensor_operations.py       # Tensor creation, shapes, math, GPU
│   └── autograd_demo.py           # Automatic differentiation & gradients
│
├── 📂 02_neural_networks/
│   ├── simple_nn.py               # Feedforward NN on Iris dataset
│   └── deep_nn.py                 # Deep NN + BatchNorm + Dropout (MNIST)
│
├── 📂 03_convolutional_networks/
│   ├── cnn_image_classifier.py    # CNN for CIFAR-10 image classification
│   └── transfer_learning.py       # Fine-tuning ResNet18
│
├── 📂 04_recurrent_networks/
│   ├── rnn_text.py                # Vanilla RNN for sequence modeling
│   └── lstm_timeseries.py         # LSTM for time-series prediction
│
├── 📂 05_training_utils/
│   ├── training_loop.py           # Reusable training pipeline + early stopping
│   └── save_load_model.py         # Model checkpointing & resuming
│
└── README.md
🚀 Getting Started
1. Clone the Repository
bashgit clone https://github.com/YOUR-USERNAME/deep-learning.git
cd deep-learning
2. Create a Virtual Environment
bash# Using venv
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Or using conda
conda create -n deeplearning python=3.10
conda activate deeplearning
3. Install Dependencies
bashpip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib scikit-learn

💡 For CPU-only installation: pip install torch torchvision torchaudio

4. Run Any Program
bashpython 01_basics/tensor_operations.py
python 02_neural_networks/simple_nn.py
python 03_convolutional_networks/cnn_image_classifier.py

📚 Programs Overview
#FileConceptDatasetAccuracy1tensor_operations.pyTensors, shapes, math ops——2autograd_demo.pyGradients, backprop, autograd——3simple_nn.pyFeedforward Neural NetworkIris~97%4deep_nn.pyDeep NN + BatchNorm + DropoutMNIST~98%5cnn_image_classifier.pyConvolutional Neural NetworkCIFAR-10~82%6transfer_learning.pyFine-tuning ResNet18CIFAR-10~90%+7rnn_text.pyVanilla RNNSequences—8lstm_timeseries.pyLSTM Time-SeriesSine WaveMSE < 0.0019training_loop.pyTraining pipeline + early stopMNIST~98%10save_load_model.pyCheckpointing & resuming——

🧩 Topics Covered
🔷 Fundamentals

Tensor creation, indexing, reshaping, broadcasting
Automatic differentiation with autograd
GPU acceleration with CUDA

🔷 Neural Networks

Fully connected / feedforward networks
Activation functions: ReLU, Sigmoid, Tanh, Softmax
Loss functions: CrossEntropy, MSE, BCE
Optimizers: SGD, Adam, AdamW, RMSProp

🔷 Regularization

Dropout
Batch Normalization
Weight Decay / L2 Regularization
Early Stopping

🔷 Convolutional Neural Networks

Conv2D, MaxPooling, Padding
Image classification pipeline
Data augmentation (RandomCrop, Flip, Normalize)
Transfer Learning with pretrained ResNet

🔷 Recurrent Networks

Vanilla RNN
Long Short-Term Memory (LSTM)
Sequence-to-one prediction
Time-series forecasting

🔷 Training Best Practices

Learning rate schedulers (StepLR, CosineAnnealing, ReduceOnPlateau)
Model checkpointing
Train / Validation / Test split
Gradient clipping


📈 Learning Path
📌 Start Here
     │
     ▼
┌─────────────────────┐
│  Tensors & Autograd │  ← 01_basics/
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Simple Neural Net  │  ← 02_neural_networks/simple_nn.py
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Deep NN + Dropout  │  ← 02_neural_networks/deep_nn.py
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  CNN (Images)       │  ← 03_convolutional_networks/
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Transfer Learning  │  ← 03_convolutional_networks/transfer_learning.py
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  RNN → LSTM         │  ← 04_recurrent_networks/
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Training Pipeline  │  ← 05_training_utils/
└─────────────────────┘
     │
     ▼
🎯 You're Ready for Advanced DL!

🛠️ Requirements
txttorch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
numpy>=1.23.0
matplotlib>=3.6.0
scikit-learn>=1.2.0
Install all at once:
bashpip install -r requirements.txt

💡 Key Concepts Cheat Sheet
TermDescriptionTensorMulti-dimensional array (PyTorch's core data structure)AutogradAutomatic gradient computation for backpropagationEpochOne full pass through the training datasetBatch SizeNumber of samples processed before updating weightsLearning RateStep size for gradient descent optimizationOverfittingModel memorizes training data, fails on new dataDropoutRandomly disables neurons during training to prevent overfittingBatchNormNormalizes layer inputs to stabilize and speed up trainingCNNUses convolutional filters to detect spatial patterns in imagesLSTMRNN variant that handles long-range dependencies in sequences

📖 References & Resources

📘 PyTorch Official Documentation
📗 Deep Learning Book — Goodfellow et al.
📙 CS231n — CNNs for Visual Recognition (Stanford)
📕 fast.ai Practical Deep Learning
📓 PyTorch Tutorials


🤝 Contributing
Contributions, suggestions, and improvements are welcome!

Fork the repository
Create a new branch: git checkout -b feature/add-transformer
Commit your changes: git commit -m 'Add transformer implementation'
Push: git push origin feature/add-transformer
Open a Pull Request


📄 License
This project is licensed under the MIT License — see the LICENSE file for details.

🙌 Author
Rajan Upadhyay
If this helped you, please ⭐ star the repo!

<div align="center">
  <sub>Built with PyTorch 🔥 | Keep Learning 🚀</sub>
</div>
