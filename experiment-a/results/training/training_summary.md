# Training Summary

This directory contains training diagnostics for representative models from the ensemble of 20 trained networks.

## Training Dynamics

All models were trained using the Adam optimizer with the following configuration:
- **Learning rate**: 0.001
- **Batch size**: 32
- **Epochs**: 50
- **Training trials**: 500

## Included Training Plots

Representative training curves are provided for selected models (seeds: 1, 5, 10, 15, 20) to demonstrate typical training behavior:

- `sys_A{seed}_err.png`: Error (loss) history
- `sys_A{seed}_param.png`: Gradient magnitude history

## Interpreting the Plots

### Error History (e.g., [`sys_A1_err.png`](https://github.com/lorenapuhl/Backengineering-neural-networks/blob/main/experiment-a/results/training/sys_A1_err.png))

The error history shows the mean MSE loss after each weight update:

**Loss Definition**: MSE = (z(t) - target(t))² averaged over all trials in a batch

**Typical Behavior**:
- **Initial phase** (0-200 steps): Rapid decrease in loss from ~0.8 to ~0.05
- **Convergence phase** (200-1600 steps): Gradual decrease and stabilization near 0.0
- **Final loss**: Typically converges to < 0.01

The sharp initial drop indicates the network quickly learns the basic task structure, while the gradual convergence reflects fine-tuning of the integration and memory mechanisms.

### Gradient History (e.g., [`sys_A1_param.png`](https://github.com/lorenapuhl/Backengineering-neural-networks/blob/main/experiment-a/results/training/sys_A1_param.png))

The gradient history shows the gradient norm |∂ε/∂θ| for each parameter group after each update:

**Parameters**:
- **Parameter 1** (blue): Input weights **I** 
  - Gradient norm: |∂ε/∂**I**| = √(Σᵢ₌₁ᴺ (∂ε/∂Iᵢ)²)
- **Parameter 2** (orange): Output weights **W**
  - Gradient norm: |∂ε/∂**W**| = √(Σᵢ₌₁ᴺ (∂ε/∂Wᵢ)²)

**Typical Behavior**:
- **Initial phase** (0-200 steps): 
  - Parameter 2 (W) shows very large gradients (~10) indicating strong learning signal for output weights
  - Parameter 1 (I) shows moderate gradients (~1-2)
- **Convergence phase** (200-1600 steps): Both parameters show decreasing gradient magnitudes approaching 0
- **Relative magnitudes**: Output weights (W) consistently show larger gradients than input weights (I), suggesting the output mapping is the primary learning challenge

The rapid decrease in gradient norms indicates successful convergence and stable optimization.

## Training Consistency

Across all 20 models, training exhibits similar characteristics:
- Consistent convergence within 50 epochs
- No signs of overfitting or instability
- Final loss values vary based on random initialization (see `../performance/` for comparative analysis)

## Convergence Criteria

Models are considered converged when:
1. Loss plateaus (change < 0.001 over 100 steps)
2. Gradient norms decrease below 0.1
3. 50 epochs completed

All saved models meet these criteria.
