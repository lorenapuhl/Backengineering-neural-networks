# Results

This directory contains analysis outputs from trained RNN models.

## Directory Structure

### `performance/`
Contains cross-model comparative analysis:
- [`sys_A_performance.png`](https://github.com/lorenapuhl/Backengineering-neural-networks/blob/main/experiment-a/results/performance/sys_A_performance.png): Scatter plots showing relationships between task precision and architectural features (weight correlation, participation ratios) across 20 trained models
- [`performance_analysis.md`](https://github.com/lorenapuhl/Backengineering-neural-networks/blob/main/experiment-a/results/performance/performance-analysis.md): Detailed interpretation of results and findings

### `training/`
Contains training diagnostics for selected representative models:
- `sys_A{seed}_err.png`: Loss (MSE) history during training
- `sys_A{seed}_param.png`: Gradient magnitude history for trainable parameters

### `model_results.txt`
For each of the trained models `sys_A{seed}`for seeds 1 to 20, the files coontains the numerical values for:

- **Precision**: MSE-Loss between network output and target function
- **Weight correlation** between trained input-weights $\mathbf{I}$ and $\mathbf{W}$
- **I-participation ratio**: Participation ratios of weights $\mathbf{I}$ with the Principal Component axes of the intrinsic network dynamics
- **w-participation ratio**: Participation ratios of weights $\mathbf{W}$ with the Principal Component axes of the network dynamics during memory (phase, during which the output features a plateau)

## Generating Results

### Performance Analysis
Run the analysis section in `main.py` with models 1-20 loaded:

```python
ana = 0  # Set to 1 to enable full analysis
```

This will generate the `sys_A_performance.png`plot.

### Training Curves
Training plots are automatically generated during training for each model. To regenerate:

```python
train = 1  # Enable training mode
```

## Interpreting Results

### Performance Plot
- **X-axis**: Task precision (lower is better) - MSE between network output and target
- **Y-axis**: Architectural features
  - **Blue points**: Weight correlation between I and w
  - **Green points**: Input participation ratio (alignment with spontaneous activity PCs)
  - **Red points**: Output participation ratio (alignment with memory period PCs)


### Training Curves
- **Loss plot**: Should show monotonic decrease; plateaus indicate convergence
- **Gradient plot**: Multiple lines for different parameter groups (I and w); decreasing magnitude indicates convergence

## Statistical Analysis

For 20 independent models:
- Mean precision: MSE-Loss $= 0.0194$
- Standard deviation: $\sigma = 0.009$
