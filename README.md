# Backengineering-neural-networks
Master's thesis in neurophysics at ENS, Paris

Understanding the neuronal firing-rate patterns of the brain poses a significant challenge. An increasingly popular approach involves simulating experimental observations using artificial recurrent neural networks, which are more convenient and share key characteristics with their biological counterparts. We train recurrent neural networks (RNNs) on biological integration and memory tasks, which refer to the brain's ability to accumulate incoming signals over time (integration) and to retain that information (memory). By reverse-engineering our trained RNNs, we analyze the network's structure and dynamics to uncover underlying principles. Understanding the network's dynamical solutions found by training may provide feasible hypotheses for the working mechanisms of the brain.

<p align="center">
  <img src="https://github.com/user-attachments/assets/54c92a1e-f4bf-4d6c-9990-5ef206f5dea7" alt="RNN" width="70%"/>
</p>

Let's dive in into some concepts using physics jargon.

´We began by analyzing the dynamics (behaviour over time) of each neuron's firing-rate (how "active" it is), and found, that chaotic dynamics yields the highest computuaional power (how complex the problems are, which a RNN can solve)
<p align="center">
  <img src="https://github.com/user-attachments/assets/4970e261-344f-4819-bd55-e0c626b607e9" alt="firing rate dynamics" width="70%"/>
</p>

We proceeded by visualising firing-rate trajectories in Principal Component-space (the space, in which the most important parts of the firing rate dynamics can be seen), to obtain valuable insights on how our RNN its given problem.


Ultimately, we attempted at consolidating our findings, by proposing an analytical solution for the output-weights. We correlated our results with trained output-weights to estimate their accuracy. While we did not obtain a satisfying value, the standard deviation of our results was comparably small. We concluded, that while our analytical expression was not exact, it still captured certain systematic aspects of our RNN’s integration- and memory-mechanisms.
