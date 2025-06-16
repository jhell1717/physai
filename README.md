# PhysicsNemo-Learning

A personal research and experimentation repository for exploring **Physics-Informed Machine Learning (PIML)** using the [NVIDIA PhysicsNemo](https://developer.nvidia.com/physicsnemo) framework.

This project is aimed at testing and learning the implementation of physics-based neural networks using PhysicsNemo with a focus on partial differential equations (PDEs), simulation-based inference, and neural operators.

---

## 📁 Project Structure
```plaintext
physai/
│
├── data/ # Datasets or simulation output files
│ └── example_dataset/ # Sample datasets (e.g., fluid dynamics, heat transfer)
│
├── notebooks/ # Jupyter notebooks for experiments and visualisations
│ └── 01_intro_physicsnemo.ipynb
│
├── configs/ # Config files for experiments
│ └── config_pinn.yaml
│
├── scripts/ # Training, evaluation, and data generation scripts
│ └── train_pinn.py
│
├── environment.yml # Conda environment file
├── .gitignore
└── README.md # Project overview
```

---

## Getting Started

### Prerequisites
- Ubuntu OS
- Python 3.8+
- NVIDIA GPU (see NVIDIA recommended)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [PhysicsNemo](https://developer.nvidia.com/physicsnemo)

### Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/physicsnemo-learning.git
cd physicsnemo-learning

# (Optional) Create and activate a conda environment
conda create -n physicsnemo python=3.10
conda activate physicsnemo

# Install dependencies
pip install -r requirements.txt
