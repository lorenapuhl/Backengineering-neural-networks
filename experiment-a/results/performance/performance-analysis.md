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
- This alignment appears to be a general feature of the dynamics rather than a performance-differentiating factor.


**Interpretation**:
- **No clear correlation between I-participation ratio and performance**
- The wide range of I-part.ratio values (~305-342) suggests all networks use relatively high-dimensional input representations
- The distribution of input weights across principal components does not appear to be the primary determinant of task success
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

### Primary Finding
**None of the measured architectural features (weight correlation, I-participation ratio, or W-participation ratio) show meaningful correlation with task performance.** 

This surprising result suggests that:
1. Performance differences arise from features not captured by these global metrics
2. The network solution is robust to variations in these architectural properties
3. More subtle or dynamical properties may be responsible for performance differences

### Mechanistic Insights

1. **Universal Architectural Features**: 
   - All networks converge to similar weight correlations (~0.82-0.89)
   - All networks use high-dimensional representations (I-part.ratio ~305-342, W-part.ratio ~307-331)
   - These appear to be general features of the learned solution

2. **Hidden Performance Determinants**:
   - Performance variation (3.2×) must arise from factors not measured here
   - Possibilities include: fine-scale weight structure, dynamical properties, initial conditions, precise timing of neural responses

3. **Robustness of Solution Space**:
   - Multiple architectural configurations within the observed ranges lead to successful solutions
   - The task may have a large basin of attraction in weight space

### Implications for Network Design

- **Random initialization still matters**: Despite similar architectures, performance varies 3.2× across seeds
- **Global metrics insufficient**: Standard measures (correlation, participation ratios) don't predict quality
- **Need for finer analysis**: Future work should examine:
  - Temporal dynamics and trajectories
  - Fine-scale weight structure
  - Network response properties during specific task phases
  - Higher-order statistical properties

## Recommendations for Future Work

1. **Dynamical analysis**: Examine neural trajectories during task execution to identify performance-critical dynamics
2. **Weight substructure**: Analyze specific patterns within **I** and **W** rather than global statistics
3. **Timing precision**: Measure temporal accuracy of integration and memory maintenance
4. **Principal component analysis**: Investigate which specific PCs are engaged, not just how many
5. **Perturbation studies**: Test sensitivity of high vs. low performers to weight perturbations
6. **Alternative metrics**: Explore measures like effective dimensionality during task execution, separability of representations, or dynamical stability

## Revised Hypotheses

Given the null results for the original hypotheses:

**Original Hypothesis (Not Supported)**: Networks that distribute input across multiple PCs perform better
**Revised Hypothesis**: All successful networks use high-dimensional representations; performance differences arise from temporal precision or fine-scale weight organization

**Original Hypothesis (Not Supported)**: I-W correlation predicts performance  
**Revised Hypothesis**: I-W correlation is a universal feature of trained networks; other alignment properties (e.g., with specific task-relevant subspaces) may matter

## Data Access

Raw data for this analysis:
- Model files: `../../models/sys_A{1-20}_state.pth`
- Analysis script: `../../main.py` (set `ana = 1`)
- Figure generation: Scatter plot created by analyzing all 20 models
