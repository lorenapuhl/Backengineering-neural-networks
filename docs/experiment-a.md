# RNN Memory Dynamics – System A  

This project implements and analyzes a recurrent neural network (RNN) designed to **simulate memory-like behavior** from temporal input signals. The model is trained to respond to signals of varying onset and duration and then evaluated for its performance and internal structure.  

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


