# PFA PyTorch Implementation

A complete PyTorch implementation of Projected Federated Averaging (PFA) with Differential Privacy, fully aligned with the original TensorFlow version.

## 🎯 Features

- **Complete Algorithm Support**: FedAvg, PFA, DP-FedAvg, DP-PFA
- **TensorFlow Alignment**: All implementations match the original TensorFlow version
- **Differential Privacy**: Full DP support with privacy accounting
- **Flexible Architecture**: Easy to extend and modify

## 📁 Project Structure

```
pfa_pytorch/
├── src/
│   ├── algorithms/          # Federated learning algorithms
│   │   ├── fedavg.py       # FedAvg implementation
│   │   ├── pfa_tf.py       # PFA (TensorFlow aligned)
│   │   ├── dp_fedavg_tf.py # DP-FedAvg (TensorFlow aligned)
│   │   └── dp_pfa.py       # DP-PFA implementation
│   ├── models/              # Model architectures
│   │   └── cnn.py          # CNN model (TensorFlow aligned)
│   ├── data/                # Data handling
│   │   ├── datasets.py     # Dataset wrappers
│   │   └── federated.py    # Federated data splitting
│   ├── privacy/             # Privacy utilities
│   │   ├── accountant.py   # Privacy accounting
│   │   └── noise.py        # Noise management
│   └── utils/               # Utility functions
│       └── lanczos.py      # Lanczos projection
├── main_tf_aligned.py       # Main script (TensorFlow aligned)
├── test_simple_aligned.py   # Test suite
└── run_tf_aligned_example.py # Example usage
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pfa_pytorch.git
cd pfa_pytorch
```

2. Install dependencies:
```bash
pip install torch torchvision numpy scipy
```

### Basic Usage

#### Run FedAvg
```bash
python main_tf_aligned.py --algorithm fedavg --N 10 --max_steps 1000
```

#### Run PFA
```bash
python main_tf_aligned.py --algorithm pfa --projection --proj_dims 5 --N 10
```

#### Run DP-FedAvg
```bash
python main_tf_aligned.py --algorithm dp_fedavg --dpsgd --eps gauss1 --N 10
```

#### Run DP-PFA
```bash
python main_tf_aligned.py --algorithm dp_pfa --dpsgd --projection --proj_dims 5
```

### Example Usage

```python
from src.models.cnn import MNISTCNN
from src.algorithms.fedavg import FedAvg
from src.algorithms.pfa_tf import PFA_TF

# Create model
model = MNISTCNN()

# FedAvg
fedavg = FedAvg(model, lr=0.1)
fedavg.local_update(dataset, local_steps=100, batch_size=4)

# PFA
pfa = PFA_TF(model, lr=0.1, proj_dims=5)
pfa.set_public_clients(epsilons)
pfa.aggregate(client_id, updates, is_public=True)
```

## 🔧 Key Features

### TensorFlow Alignment
- **Model Architecture**: Exact CNN structure match
- **Training Loop**: Uses `local_steps` instead of `epochs`
- **Sampling Strategy**: Matches TensorFlow client sampling
- **Aggregation Weights**: Implements PFA public/private client classification
- **Noise Calculation**: Uses TensorFlow's simplified formula

### Algorithms Implemented
- **FedAvg**: Standard federated averaging
- **PFA**: Projected federated averaging with Lanczos projection
- **DP-FedAvg**: Differential privacy with FedAvg
- **DP-PFA**: Differential privacy with PFA
- **WeiAvg**: Weighted averaging based on privacy budgets

### Privacy Features
- **Privacy Accounting**: Tracks epsilon and delta consumption
- **Gradient Clipping**: L2 norm clipping for DP
- **Noise Addition**: Gaussian noise for privacy protection
- **Budget Management**: Prevents privacy budget exhaustion

## 📊 Performance

The implementation maintains the same performance characteristics as the original TensorFlow version while providing the flexibility and ease of use of PyTorch.

## 🧪 Testing

Run the test suite:
```bash
python test_simple_aligned.py
```

Run the example:
```bash
python run_tf_aligned_example.py
```

## 📝 Citation

If you use this code in your research, please cite the original PFA paper and mention this PyTorch implementation.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

For questions or issues, please open an issue on GitHub.
