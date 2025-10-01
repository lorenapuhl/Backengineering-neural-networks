# Backengineering-neural-networks
Master's thesis in neurophysics at ENS, Paris


# **1. Motivation**


Understanding the neuronal firing-rate patterns of the brain poses a significant challenge. An increasingly popular approach involves simulating experimental observations using artificial recurrent neural networks, which are more convenient and share key characteristics with their biological counterparts. We train recurrent neural networks (RNNs) on biological integration and memory tasks, which refer to the brain's ability to accumulate incoming signals over time (integration) and to retain that information (memory). By reverse-engineering our trained RNNs, we analyze the network's structure and dynamics to uncover underlying principles. Understanding the network's dynamical solutions found by training may provide feasible hypotheses for the working mechanisms of the brain.

<p align="center">
  <img src="https://github.com/user-attachments/assets/45b3f5d8-692c-4bbf-af47-5ac4bb128696" alt="brain" height="250" style="margin-right:10px;">
  </p>
  
<p align="center">
  <img src="https://github.com/user-attachments/assets/54c92a1e-f4bf-4d6c-9990-5ef206f5dea7" alt="RNN" height="250">
  </p>
  
<p align="center">
  <em>Figure 1: (left) Measuring the brain's electrical signals. (right) Simulating the brain's electrical signals using a Recurrent Neural Network. It consists of input-weights I feeding signals to the recurrent net G. The output is read out using the weights W</em>
</p>

# **2. Project Overview**


## **2.1 Training a RNN**


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

## **2.2 Reverse-Engineering and Analysis**

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


## **2.3 Conclusion**

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
