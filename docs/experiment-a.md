# RNN Memory Dynamics – System A  

This project trains a recurrent neural network (RNN) to solve the biological integration and memory task for inputs of different durations. Model A is implemented by training both - input weights $\mathbf{I}$ and output weights $\mathbf{W}$. A set of twenty different models are trained to perform a statistical analysis of the dynamical solutions after training. The analysis is conducted by measuring the network's precision in solving the integration and memory tasks. Furthermore, we make our first attempt at reverse-engineering the network's solution found by training. In other words, we are trying to fin out according to which principles the training-based solution solves the denoted integration- and memory-problem; does the network leverage certain dynamical axes (so-called *Principal Components*) to solve the task ? If so, then the *participation ratios* (a number showing how aligned the $\mathbf{I}$- and $\mathbf{W}$-weights are with the denoted axes), should be correlated with the network's performance. Equally, we ask ourselves, if the solution involves some finetuning between input- and output-weights (a so-called *correlation* between the weights). If so, then plotting the denoted propterties with respect to the network performance should reveal a distinct pattern.


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


