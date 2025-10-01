# RNN Memory Dynamics – System A  

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

## Features  

- **Custom RNN architecture** (`Net1` from `utils.py`)  
- **Flexible parameters** for signal generation and network training  
- **Training mode** with multiple random seeds  
- **Analysis mode** for precision, weight correlation, and participation ratios  
- **Visualization** of performance metrics  

---

## Parameters  

- **Network**  
  - Hidden size: 500  
  - Time constant: 100 ms  
  - Gain: 1.5  
  - Initialization seed configurable  

- **Data**  
  - Signal onset: ~3000 ms (with noise)  
  - Signal duration: average 150 ms  
  - Strength: 2.0  
  - Training trials: 500  

- **Training**  
  - 50 epochs  
  - Batch size: 32  
  - Learning rate: 0.001  

---


