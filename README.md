# Deep Reinforcement Learning Project: Double Deep Q-Network (DDQN) with Learnable Value Estimation

This repository contains a deep reinforcement learning project focusing on implementing the Dueling Deep Q-Network (DDQN) algorithm with a learnable value estimation component. 

## Overview

Deep Q-Networks (DQNs) have shown remarkable success in learning to play various games directly from raw pixel inputs. However, traditional DQNs suffer from overestimation bias, which can lead to suboptimal performance. The Double DQN (DDQN) algorithm addresses this issue by decoupling the action selection from the value estimation, resulting in more stable and accurate Q-value predictions.

In this project, we extend the DDQN algorithm by incorporating a learnable value estimation component. This component aims to dynamically adjust the value estimates based on the agent's experience, leading to improved performance and faster convergence.

