# RNN Memory Dynamics – System A 
## Motivation

This project trains a recurrent neural network (RNN) to solve a biological integration and memory task with inputs of varying durations. Model A is implemented by training both the input weights $\mathbf{I}$ and the output weights $\mathbf{W}$. To enable statistical analysis of the resulting dynamical solutions, a set of twenty independent models are trained.

The analysis focuses on measuring the network’s precision in performing the integration and memory tasks. Beyond evaluating performance, we also attempt a first step toward reverse-engineering the network’s learned solution. In other words: according to which principles does the trained network solve the integration and memory problem?

One hypothesis is that the network leverages specific dynamical axes —*Principal Components*— to carry out the task. If this is the case, then the *participation ratios* (which quantify the alignment between $\mathbf{I}$ and $\mathbf{W}$ and the identified axes) should correlate with task performance. Another possibility is that the solution relies on fine-tuned interactions between input and output weights—that is, a *correlation* between them. If so, plotting these properties against network performance should reveal characteristic patterns.

<p align="center">
  <img src="https://github.com/user-attachments/assets/5ec6fa8d-5bbb-4a3b-bcda-cfd13e1f04d8" alt="pca" width="50%">
</p>
<p align="center">
  <em>Figure 1: Schematic understanding of Principal Component Analysis (PCA). The figure shows the axes (or directions) of highest movement, knwon as Principal Components. (Figure from https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html) </em>
</p>

---

## Overview

- Train an RNN on inputs of different durations to perform integration and memory tasks.  
- Evaluate the network's precision in solving the task
- Test whether task performance is linked to:  
  - **Participation ratios**: alignment of input (`I`) and output (`W`) weights with the principal axes.  
  - **Weight correlations**: fine-tuning between input and output weights.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- utils.py

## Network Architecture

- **Input size**: 1
- **Hidden size**: 500 neurons
- **Output size**: 1
- **Time constant (τ)**: 100 ms
- **Time step (dt)**: 1 ms
- **Connection strength (g_G)**: 1.5

## Dataset Parameters

- **Training trials**: 500
- **Total time steps**: 7000
- **Signal onset time**: 3000 ms (±1000 ms deviation)
- **Signal duration**: 150 ms (±100 ms deviation)
- **Input signal strength**: 2.0
- **Cost evaluation onset**: 4500 ms

## Training Parameters

- **Epochs**: 50
- **Batch size**: 32
- **Learning rate**: 0.001
- **Optimizer**: Adam

## Usage

### Training Models

Set `train = 1` in the script to train 20 independent models with different random seeds:

```python
train = 1  # Enable training
```

This will train models with seeds 1-21 and save them as `sys_A{seed}_state.pth`.

### Analysis

The script performs comprehensive analysis on trained models:

1. **Single Model Evaluation**: Loads a specific model and evaluates performance
2. **Multi-Model Analysis**: Analyzes all 20 models (seeds 1-20) and computes:
   - Precision across different signal durations (50, 150, 300 ms)
   - Weight correlation between input and output weights
   - Input weight participation ratio
   - Output weight participation ratio

### Running the Analysis

```python
ana = 0  # Set to 1 to enable analysis
```

The analysis evaluates networks on three signal durations:
- 50 ms (short)
- 150 ms (medium)
- 300 ms (long)

## Output

### Saved Models
- Model states saved as: `sys_A{seed}_state.pth`

### Analysis Plots
- **Performance analysis plot**: `sys_A_performance.png`
  - Shows relationships between precision and:
    - Weight correlation (blue)
    - Input participation ratio (green)
    - Output participation ratio (red)

## Key Functions (from SYS_A_modules)

- `Net1()`: RNN model class
- `evaluate()`: Evaluates network performance
- `precision()`: Measures task precision
- `weight_correlation()`: Computes correlation between input and output weights
- `I_partic_ratio()`: Calculates participation ratio for input weights
- `w_partic_ratio()`: Calculates participation ratio for output weights

## Model Selection

The script includes functionality to load and analyze specific models:

```python
saved_model = f"sys_A{seed}"  # Load model with specific seed
```

## Research Questions

This implementation helps investigate:

1. Do networks leverage specific dynamical axes (Principal Components) to solve the task?
2. Do participation ratios correlate with task performance?
3. Are fine-tuned interactions between input and output weights critical for the solution?
4. What patterns emerge when plotting weight properties against network performance?

## Notes

- Initial hidden state (h0) is set to zero by default
- The threshold parameter increases at a rate of 0.007
- Training and analysis can be toggled independently via `train` and `ana` flags

