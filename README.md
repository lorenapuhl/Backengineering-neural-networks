# Backengineering-neural-networks
Master's thesis in neurophysics at ENS, Paris

## **1. Motivation**

Understanding the neuronal firing-rate patterns of the brain poses a significant challenge. An increasingly popular approach involves simulating experimental observations using artificial recurrent neural networks, which are more convenient and share key characteristics with their biological counterparts. We train recurrent neural networks (RNNs) on biological integration and memory tasks, which refer to the brain's ability to accumulate incoming signals over time (integration) and to retain that information (memory). By reverse-engineering our trained RNNs, we analyze the network's structure and dynamics to uncover underlying principles. Understanding the network's dynamical solutions found by training may provide feasible hypotheses for the working mechanisms of the brain.

<p align="center">
  <img src="https://github.com/user-attachments/assets/45b3f5d8-692c-4bbf-af47-5ac4bb128696" 
       alt="brain" height="250", style="margin-right:20px;">
  <img src="https://github.com/user-attachments/assets/54c92a1e-f4bf-4d6c-9990-5ef206f5dea7" 
       alt="RNN" height="250">
</p>

<p align="center">
  <em>Figure 1: (left) Measuring the brain's electrical signals. (right) Simulating the brain's electrical signals using a Recurrent Neural Network. It consists of input-weights I feeding signals to the recurrent net G. The output is read out using the weights W.</em>
</p>


## **2. Project Overview**

### **2.1 Training a RNN**

Let's very briefly dive into some of our project's proceedings using minimal physics jargon.

We began by training a series of recurrent neural networks (RNNs) to address the biological integration and memory problem. In this task, the network receives constant input signals of varying durations. To succeed, the RNN must measure for how long the input was active (integration) and represent this information through a sustained output signal (memory). The amplitude (signal height) of the output encodes the duration of the corresponding input. Training consisted of adjusting both the input and output weights of the network.

<p align="center">
  <img src="https://github.com/user-attachments/assets/22ca8b96-d2f7-4dbe-af2f-ddc66c82608b" alt="data set" width="70%">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/7ec6c884-6606-47d7-8007-6c99829af0c0" alt="network output" width="70%">
</p>

<p align="center">
  <em>Figure 2: (top) Data-set featuring integration and memory patterns. The figure illustrates input-signals of different durations. The according target-output is a constant signal encoding the according input-duration through its height. (bottom) An RNN's typical output after having been trained on such a data-set.</em>
</p>

### **2.2 Reverse-Engineering and Analysis**

We proceeded by visualising firing-rate trajectories in Principal Component-space (the space, in which the most important parts of the firing rate dynamics can be seen), to obtain valuable insights on how our RNN solves its given problem. We found, that input-signals of different durations lead to parallel firing-rate trajectories (the path the dynamics traces in PC-space), while their separating distances were correlated to the according signal-times. We therefore hypothesized, that integration relied on measuring the denoted distances. On the other hand, we suggested that output-weights were fine-tuned to rule out the remaining firing-rate dynamics, in order to output a plateau.

<p align="center">
  <img src="https://github.com/user-attachments/assets/467b5ae7-b3d3-4a1c-97ce-d3d2beb9503c" 
       alt="trajectories" 
       width="75%">
</p>

<p align="center">
  <em>Figure 3: Three different trajectories r(t) for inputs of durations 50ms (blue and cyan), 150ms (green and olive) and 300ms (red and magenta) in Principal Component Space.  
  (a) Time-frame where the input is turned on. The input-weights I are illustrated using a vector-arrow.  
  (b) Intrinsic phase, where the network processes the input before entering plateau-phase. The latter denotes the time-frame where the network exhibits a plateau-like output to mimic memory behaviour.  
  (c) Entire trajectory. Blue, green and red represent r(t) during the input-dominated and intrinsic phases. Cyan, magenta and olive are used once the curves enter the memory-period or plateau-phase, and are labelled as "memory".</em>
</p>

### **2.3 Conclusion**

Ultimately, we attempted at consolidating our findings. We proposed an analytical solution for how our RNN is able to solve integration- and memory behaviours observed in neuroscience.

<p align="center">
  <img src="https://github.com/user-attachments/assets/0fd80836-9122-48cb-9384-c56797fff4b9" alt="equation1" width="60%">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/5a379490-cd11-4441-87a3-8ca2f309c4ce" alt="equation2" width="60%">
</p>

<p align="center">
  <em>Figure 4: (top) Basic structure of our proposed equation. (bottom) Determining each component.</em>
</p>

## **3. Repository Structure**

```
├── README.md                    # This file - project overview
├── Thesis.pdf                   # Complete master's thesis document
├── experiment-a/                # Training both input (I) and output (W) weights
│   ├── main.py                  # Training and analysis script
│   ├── SYS_A_modules.py        # Network architecture and utilities
│   ├── requirements.txt         # Python dependencies
│   ├── models/                  # Trained model weights
│   └── results/                 # Training curves and performance analysis
└── experiment-b/                # Coming soon: Alternative training configurations
```

### Experiment-A: Joint Training of Input and Output Weights

This experiment trains RNNs where both input weights (**I**) and output weights (**W**) are optimized simultaneously. Twenty independent models were trained to enable statistical analysis of learned solutions. The analysis investigates whether performance correlates with architectural features such as weight correlations and participation ratios. See [`experiment-a/README.md`](https://github.com/lorenapuhl/Backengineering-neural-networks/tree/main/experiment-a) for detailed documentation.


### Future Experiments

Additional experiments exploring alternative training configurations and analysis strategies will be added soon, including:
- **Experiment-B**: Training only output-weights, while input-weights randonly initialized. Comparing performances with results of Experiment A. Visualising firing-rate trajectories in Principal Component - Space
- **Experiment-C**: Training only output-weights, while input-weights are manually set along specific Principal Components.
- **Experiment-D**: Training only input-weights, while output-weights are manually set along specific Principal Components. Furthermore, we also choose to set input-weights along specific axes to understand which initial conditions facilitate training.
- **Experiment-E**: Experimenting with different network sizes and exloring their degrees of dynamical freedom. We analyze the eigenvalue spectra of their respective PC-spaces and participation ratios to define their degrees of computational freedom.
- **Experiment-F**: Training only (randomly initialised) input-weights and comparing network performances between experiments A, B and F


## **4. Getting Started**

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Backengineering-neural-networks.git
cd Backengineering-neural-networks

# Install dependencies for experiment-a
cd experiment-a
pip install -r requirements.txt
```

### Running Experiments

See individual experiment directories for specific instructions. For example, to run experiment-a:

```bash
cd experiment-a
python main.py
```

## **5. Further Information**

For comprehensive details on the theoretical background, methodology, results, and discussion, please refer to [**Thesis.pdf**](https://github.com/lorenapuhl/Backengineering-neural-networks/blob/main/Thesis.pdf) included in this repository.

## **6. Contact**

lorena.puhl@protonmail.com
