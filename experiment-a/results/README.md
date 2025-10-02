# Results

This directory contains analysis outputs from trained RNN models.

## Directory Structure

### `performance/`
Contains cross-model comparative analysis:
- `sys_A_performance.png`: Scatter plots showing relationships between task precision and architectural features (weight correlation, participation ratios) across 20 trained models
- `performance_analysis.md`: Detailed interpretation of results and findings

### `training/`
Contains training diagnostics for selected representative models:
- `sys_A{seed}_err.png`: Loss (MSE) history during training
- `sys_A{seed}_param.png`: Gradient magnitude history for trainable parameters

## Generating Results

### Performance Analysis
Run the analysis section in `main.py` with models 1-20 loaded:

```python
ana = 0  # Set to 1 to enable full analysis
```

This will generate the `sys_A_performance.png` plot.

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

Look for:
- Linear trends indicating correlation between features and performance
- Clusters suggesting discrete solution strategies
- Outliers representing unique solutions

### Training Curves
- **Loss plot**: Should show monotonic decrease; plateaus indicate convergence
- **Gradient plot**: Multiple lines for different parameter groups (I and w); decreasing magnitude indicates convergence

## Statistical Analysis

For 20 independent models:
- Mean precision: [to be computed]
- Standard deviation: [to be computed]
- Correlation coefficients between features and performance: [to be computed]

See `performance/performance_analysis.md` for detailed statistical analysis.

