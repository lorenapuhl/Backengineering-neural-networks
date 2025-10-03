# Performance Analysis

This document provides detailed analysis of the relationship between network architecture and task performance across 20 independently trained models.

## Overview

The scatter plot [`sys_A_performance.png`](https://github.com/lorenapuhl/Backengineering-neural-networks/blob/main/experiment-a/results/performance/sys_A_performance.png) shows three architectural features plotted against task precision for all 20 trained models (seeds 1-20). Each model was trained with identical hyperparameters but different random initializations.

## Metrics Analyzed

### X-Axis: Performance (Precision)
- **Definition**: Mean MSE loss across three signal durations (50, 150, 300 ms)
- **Range**: ~0.012 to ~0.042
- **Interpretation**: Lower values indicate better performance (more precise integration and memory)

### Y-Axis: Architectural Features

#### 1. Weight Correlation (Blue Points)
- **Definition**: Normalized correlation between input weights **I** and output weights **W**
- **Formula**: correlation = (**I** · **W**) / (|**I**| × |**W**|)
- **Range**: ~0.85 to ~0.92
- **Interpretation**: Measures alignment between how information enters the network and how it's read out

#### Participation Ratios
The participation ratio D is calculated as:

```
D = (Σᵢ₌₁ᴺ μᵢ)² / Σᵢ₌₁ᴺ μᵢ²     (30)
```

where μᵢ is the i-th eigenvalue of the covariance matrix.

**Interpretation:**
- If all eigenvalues contribute equally (i.e., μᵢ/Σⱼ₌₁ᴺ μⱼ = 1/N), the dimension estimate is D = N
- Conversely, if only one eigenvalue contributes, then D = 1
- Values between 1 and N indicate the effective number of dimensions used by the system

#### 2. I-Participation Ratio (Green Points)
- **Definition**: Participation ratio of input weights **I** onto principal components of spontaneous activity
- **Formula**: PR = (Σ cᵢ)² / Σ cᵢ², where cᵢ = eigenvector_i · **I**
- **Range**: ~0 to ~4.5
- **Interpretation**: Higher values indicate input weights align with fewer principal components; lower values indicate distributed representation

#### 3. W-Participation Ratio (Red Points)
- **Definition**: Participation ratio of output weights **W** onto principal components during memory period
- **Formula**: PR = (Σ cᵢ)² / Σ cᵢ², where cᵢ = eigenvector_i · **W**
- **Range**: ~0 to ~2.3
- **Interpretation**: Higher values indicate output weights preferentially read from fewer principal components

## Key Findings

### 1. Weight Correlation vs Performance

**Observation**: Blue points show relatively stable values (~0.85-0.92) across all performance levels with no clear trend.

**Interpretation**: 
Weight correlation remains consistently high across networks, regardless of task performance. This indicates that alignment between the input ($\mathbf{I}$) and output ($\mathbf{W}$) weights is a fundamental property reliably established during training. However, such correlations alone do not explain the performance differences observed between models, suggesting the involvement of deeper dynamical mechanisms. In the following analyses of this thesis, we will further examine the role of these correlations. They prove to be highly significant for capturing input-signal duration (integration), and by visualizing the network’s dynamics, we will also be able to highlight the geometric underpinnings and functional meaning of these correlations.

### 2. I-Participation Ratio vs Performance

**Observation**

**Interpretation**:


### 3. W-Participation Ratio vs Performance

**Observation**

**Interpretation**:

## Statistical Summary

### Performance Distribution
- **Mean precision**: ~0.027
- **Best model**: ~0.012 (seed with lowest MSE)
- **Worst model**: ~0.042 (seed with highest MSE)
- **Range**: ~3.5× difference between best and worst



## Conclusions

### Primary Finding
**Input weight distribution across principal components is the strongest predictor of task performance.** Networks that distribute input signals across multiple dynamical modes achieve better integration and memory precision.


## Data Access

Raw data for this analysis:
- Model files: `../../models/sys_A{1-20}_state.pth`
- Analysis script: `../../main.py` (set `ana = 1`)
- Figure generation: Scatter plot created by analyzing all 20 models
