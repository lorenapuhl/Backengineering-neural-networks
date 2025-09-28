# Backengineering-neural-networks
Master's thesis in neurophysics at ENS, Paris

Understanding the neuronal firing-rate patterns of the brain poses a significant challenge. An increasingly popular approach involves simulating experimental observations using artificial recurrent neural networks, which are more convenient and share key characteristics with their biological counterparts. We train recurrent neural networks (RNNs) on biological integration and memory tasks, which refer to the brain's ability to accumulate incoming signals over time (integration) and to retain that information (memory). By reverse-engineering our trained RNNs, we analyze the network's structure and dynamics to uncover underlying principles. Understanding the network's dynamical solutions found by training may provide feasible hypotheses for the working mechanisms of the brain.

<p align="center">
  <img src="https://github.com/user-attachments/assets/45b3f5d8-692c-4bbf-af47-5ac4bb128696" alt="brain" width="45%" style="margin-right: 10px;"/>
  <img src="https://github.com/user-attachments/assets/54c92a1e-f4bf-4d6c-9990-5ef206f5dea7" alt="RNN" width="45%"/>
</p>
Let's very briefly dive in into some concepts using physics jargon.

´We began by analyzing the dynamics (behaviour over time) of each neuron's firing-rate (how "active" it is), and found, that chaotic dynamics yields the highest computational power (how complex the problems are, which a RNN can solve)

<p align="center">
  <img src="https://github.com/user-attachments/assets/4970e261-344f-4819-bd55-e0c626b607e9" 
       alt="firing rate dynamics" 
       width="50%" 
       height="50%"/>
</p>

We proceeded by visualising firing-rate trajectories in Principal Component-space (the space, in which the most important parts of the firing rate dynamics can be seen), to obtain valuable insights on how our RNN its given problem.

<p align="center">
  <img src="https://github.com/user-attachments/assets/467b5ae7-b3d3-4a1c-97ce-d3d2beb9503c" 
       alt="trajectories" 
       width="50%" 
       height="50%"/>
</p>

Ultimately, we attempted at consolidating our findings. We proposed an analytical solution for how our RNN is able to solve integration- and memory behaviours observed in neuroscience.
