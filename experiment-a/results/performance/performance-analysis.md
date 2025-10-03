# Performance Analysis

This document provides detailed analysis of the relationship between network architecture and task performance across 20 independently trained models.

## Overview

The scatter plot [`sys_A_performance.png`](https://github.com/lorenapuhl/Backengineering-neural-networks/blob/main/experiment-a/results/performance/sys_A_performance.png) shows three architectural features plotted against task precision for all 20 trained models (seeds 1-20). Each model was trained with identical hyperparameters but different random initializations. [`sys_A_performance_zoom1.png`](https://github.com/lorenapuhl/Backengineering-neural-networks/blob/main/experiment-a/results/performance/sys_A_performance_zoom1.png) and [`sys_A_performance_zoom2.png`](https://github.com/lorenapuhl/Backengineering-neural-networks/blob/main/experiment-a/results/performance/sys_A_performance_zoom2.png) show the features with higher resolution

## Metrics Analyzed

### X-Axis: Performance (Precision)
- **Definition**: Mean MSE loss across three signal durations (50, 150, 300 ms)
- **Range**: ~0.013 to ~0.042
- **Interpretation**: Lower values indicate better performance (more precise integration and memory)

### Y-Axis: Architectural Features

#### 1. Weight Correlation (Blue Points)
([`sys_A_performance_zoom2.png`](https://github.com/lorenapuhl/Backengineering-neural-networks/blob/main/experiment-a/results/performance/sys_A_performance_zoom2.png))
- **Definition**: Normalized correlation between input weights **I** and output weights **W**
- **Formula**: correlation = (**I** · **W**) / (|**I**| × |**W**|)
- **Range**: ~0.82 to ~0.89
- **Interpretation**: Measures alignment between how information enters the network and how it's read out

#### 2. I-Participation Ratio (Green Points)
([`sys_A_performance_zoom1.png`](https://github.com/lorenapuhl/Backengineering-neural-networks/blob/main/experiment-a/results/performance/sys_A_performance_zoom1.png))
- **Definition**: Participation ratio of input weights **I** onto principal components of spontaneous activity
- **Formula**: PR = (Σ cᵢ)² / Σ cᵢ², where cᵢ = eigenvector_i · **I**
- **Range**: ~305 to ~342
- **Interpretation**: If all eigenvalues contribute equally (i.e., cᵢ/Σⱼ₌₁ᴺ cⱼ = 1/N), the dimension estimate is PR = N. Conversely, if only one eigenvalue contributes, then PR = 1. Thus,  Values between 1 and N indicate the effective number of dimensions used by the system

#### 3. W-Participation Ratio (Red Points)
([`sys_A_performance_zoom1.png`](https://github.com/lorenapuhl/Backengineering-neural-networks/blob/main/experiment-a/results/performance/sys_A_performance_zoom1.png))
- **Definition**: Participation ratio of output weights **W** onto principal components during memory period
- **Formula**: PR = (Σ cᵢ)² / Σ cᵢ², where cᵢ = eigenvector_i · **W**
- **Range**: ~307 to ~331
- **Interpretation**: If all eigenvalues contribute equally (i.e., cᵢ/Σⱼ₌₁ᴺ cⱼ = 1/N), the dimension estimate is PR = N. Conversely, if only one eigenvalue contributes, then PR = 1. Thus,  Values between 1 and N indicate the effective number of dimensions used by the system

## Key Findings

### 1. Weight Correlation vs Performance

**Observation**: Blue points (weight correlation) cluster tightly near zero across the entire performance range (~0.82-0.89), with no discernible trend.

**Interpretation**: 
- Weight correlation shows **no correlation** with individual task performance
- All networks converge to similar correlation values regardless of quality
- This alignment appears to be a general feature of the learning dynamics rather than a performance-differentiating factor. As we will unerstand later in this thesis, correlating input- and output-weights is a crucial strategy to solve the biological integration and memory tasks.

### 2. I-Participation Ratio vs Performance

**Observation**: 
- Green points show clustering around 305-342 with no clear linear or systematic relationship to performance.
- All networks converge to similar correlation values regardless of quality
- This alignment appears to be a general feature of the solution-dynamics rather than a performance-differentiating factor.


**Interpretation**:
- **No clear correlation between I-participation ratio and performance**
- As we will unerstand later in this thesis, the dynamical solution for solving the biological integration and memory task does not rely on choosing input-vectors with certain properties. In fact, network performance for propotypes with random input-vectors and trained output-weights also perform reasonably well.

### 3. W-Participation Ratio vs Performance

**Observation**: Red points cluster around 307-331 with substantial overlap across all performance levels:

**Interpretation**:
- **No correlation between W-participation ratio and performance**
- Output weights show similar high-dimensional structure across all networks
- As we will understand later in this tesis, the dynamical solution for solving the biological integration and memory relies on picking certain principal axes, which are crucial for interpreting input durations and achieving a plateau-shaped output. The weight distributions on the remaining principal axes is less significant.

## Statistical Summary

### Performance Distribution
- **Mean precision**: ~0.025
- **Best model**: ~0.013 (lowest MSE)
- **Worst model**: ~0.042 (highest MSE)
- **Range**: ~3.2× difference between best and worst


## Conclusions

None of the measured architectural features (weight correlation, I-participation ratio, or W-participation ratio) show meaningful correlation with individual task performance. Thus, performance variation must arise from factors not measured here. 

However, we found that all networks converge to similar weight correlations and show a high weight-distribution across principal components. These features appear to be part of the general dynamics to solve biological integration and memory tasks, rather than a performance-distinguishing factor. We will understand later in this thesis, that the high correlation between input- and output-weights is crucial for a successfully working network. Furthermore, training output-weights $\mathbf{W}$ rather focusses on finding certain principal axes crucial for integrating input-signals and exhibiting a plateau-shaped output. The disctribution across the remaining PC's is less significant. Lastly, it seems that such a solution can be found for any random vector $\mathbf{I}$. Indeed networks only trained on output-weights and random input-weights are also able to solve the task.


## Data Access

Raw data for this analysis:
- Model files: `../../models/sys_A{1-20}_state.pth`
- Analysis script: `../../main.py` (set `ana = 1`)
- Figure generation: Scatter plot created by analyzing all 20 models
