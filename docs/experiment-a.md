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

This project trains RNNs (Model A) to perform integration and memory tasks by training both input weights (**I**) and output weights (**W**). The implementation trains 20 independent models to enable statistical analysis of the learned dynamical solutions.

The analysis investigates how trained networks solve integration and memory problems by examining:
- **Network precision** in task performance
- **Dynamical axes** (Principal Components) used by the network
- **Participation ratios** measuring alignment between weights and identified axes
- **Weight correlations** between input and output weights

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- SYS_A_modules (custom module - must be in the same directory)

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

This will train models with seeds 1-99 and save them as `sys_A{seed}_state.pth`.

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
Model states are saved as: `sys_A{seed}_state.pth`

**Note:** Due to file size constraints, only 1-2 representative trained models are included in the `models/` directory. To reproduce all 20 models, run the training script with `train = 1`.

### Training Outputs
For each trained model, two plots are generated:
- `sys_A{seed}_err.png`: Loss history during training
- `sys_A{seed}_param.png`: Gradient magnitudes for trainable parameters during training

Sample training plots for representative models are available in `results/training/`.

### Analysis Outputs
- **Performance analysis plot** (`sys_A_performance.png`): Scatter plot showing relationships between task precision and:
  - Weight correlation (blue): Alignment between input and output weights
  - Input participation ratio (green): How input weights align with spontaneous activity PCs
  - Output participation ratio (red): How output weights align with memory period PCs

This plot is located in `results/performance/` and provides insights into which architectural features correlate with better task performance.

## Key Functions (from SYS_A_modules)

### Core Classes

#### `Net1(tau, dt, input_size, output_size, hidden_size, g_G, seed, h0)`
Recurrent Neural Network model class implementing a continuous-time RNN.

**Parameters:**
- `tau`: Time constant (ms)
- `dt`: Time step size (ms)
- `input_size`: Number of input dimensions
- `output_size`: Number of output dimensions
- `hidden_size`: Number of hidden neurons
- `g_G`: Connection strength parameter
- `seed`: Random seed for weight initialization
- `h0`: If True, initializes hidden state to zero; otherwise random initialization

**Key Methods:**
- `forward(input, return_dynamics=False, return_frdynamics=False)`: Simulates network forward pass
  - Returns output signal
  - Optionally returns hidden state dynamics (`return_dynamics=True`)
  - Optionally returns firing rate dynamics (`return_frdynamics=True`)
- `train(...)`: Trains the network using Adam optimizer with custom MSE loss

**Network Architecture:**
- **Weights:**
  - `G`: Recurrent connectivity matrix (fixed, not trained)
  - `I`: Input weights (trainable)
  - `w`: Output weights (trainable)
  - `h0`: Initial hidden state (fixed)
- **Dynamics:** `h(t+dt) = h(t) + (dt/tau) * (-h(t) + g_G * tanh(h(t)) @ G.T + input @ I.T)`
- **Output:** `output = tanh(h) @ w`

### Loss Functions

#### `mse_loss1(output, target)`
Custom MSE loss with masked evaluation. Only computes loss where target ≠ -1 (evaluation period).

#### `mse_loss2(output, target)`
Custom MSE loss for precision calculation. Only computes loss where target > 0.

### Data Generation

#### `data1(net, trials, Nt, O_on, SO_on, O_off, SO_off, thresh, perc, cost_onset, sig)`
Generates training/testing datasets with variable signal timing.

**Returns:**
- `inputt`: Input signal tensor (trials × Nt × input_size)
- `targett`: Target output tensor (trials × Nt × output_size)
  - Contains -1 during non-evaluation periods
  - Contains threshold value during evaluation period

### Analysis Functions

#### `evaluate(net, dur_list, trials_test, ..., sig_onset)`
Visualizes network performance across different input durations. Plots network output, target, and input signals.

#### `precision(net, dur_list, trials_test, ..., sig_onset)`
Measures task precision by computing average MSE loss across different signal durations.

**Returns:** Mean precision score (lower is better)

#### `weight_correlation(net)`
Computes normalized correlation between input weights **I** and output weights **w**.

**Formula:** `correlation = (I · w) / (|I| * |w|)`

**Returns:** Correlation coefficient [-1, 1]

#### `signals(net, O_on, Nt, input_len, thresh, cost_onset, sig)`
Generates firing rate trajectories and network output for a single trial.

**Returns:**
- `r_time`: Firing rate trajectories (Nt × hidden_size)
- `output`: Network output signal (Nt,)

#### `pca_matrix(net, t1, t2, r, plot=False)`
Computes PCA on firing rate covariance matrix over specified time window.

**Returns:**
- `e_val`: Eigenvalues of covariance matrix
- `e_vec`: Eigenvectors (principal components)

#### `pca_trials(net, plot=False)`
Performs PCA by averaging covariance matrices across three trials with different input durations (50, 150, 300 ms).

**Returns:**
- `e_val`: Averaged eigenvalues
- `e_vec`: Averaged eigenvectors

#### `I_partic_ratio(net, ...)`
Calculates participation ratio of input weight vector **I** onto PCA eigenvectors from spontaneous activity.

**Formula:** `PR = (Σ cᵢ)² / Σ cᵢ²` where `cᵢ = eigenvector_i · I`

**Returns:** Participation ratio (higher values indicate alignment with fewer principal components)

#### `w_partic_ratio(net, ...)`
Calculates participation ratio of output weight vector **w** onto PCA eigenvectors during the memory period.

**Formula:** `PR = (Σ cᵢ)² / Σ cᵢ²` where `cᵢ = eigenvector_i · w`

**Returns:** Participation ratio

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

## License

[Add your license information here]

## Citation

[Add citation information if applicable]
