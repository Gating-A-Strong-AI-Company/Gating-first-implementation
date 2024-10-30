# Welcome to Gating

This repository presents the first implementation of Gating, a novel AI technology designed to replace traditional deep learning approaches. Gating leverages transient, dynamic connections in neural networks inspired by biological principles, allowing for the selective activation of network pathways. This mechanism enhances the adaptability and efficiency of the network, enabling more nuanced and flexible problem-solving capabilities.

To learn more, visit the official site at [www.gating.ai](https://www.gating.ai).

## What is Gating?

Gating is a new paradigm in artificial intelligence inspired by transient, selective pathways found in biological neural networks. Unlike deep learning, which relies on static weights learned through backpropagation, Gating introduces a dynamic mechanism where connections in the network can be temporarily enabled or disabled based on external triggers.

### Key Concepts

1. **Dynamic Connection Activation**:
   - Connections between neurons are not fixed; they can be selectively activated or deactivated based on context or triggers, which enables transient reconfiguration of the network.
   
2. **Biological Inspiration**:
   - Gating takes inspiration from metabotropic receptors and G protein-gated ion channels in biological systems, enabling AI models to mimic some aspects of flexible, context-driven biological neural activity.

3. **Efficient and Adaptable**:
   - Gated networks allow for more efficient memory usage, adaptability to different tasks, and potential reductions in training data and compute resources.

4. **Towards Real-time Reconfiguration**:
   - The Gating model can rewire itself during execution, making it possible for networks to adapt dynamically to changing input or environmental conditions in real-time.

### The Gating Model

In this repository, the Gating model is implemented in Python using PyTorch. The implementation includes:
- **Gate class**: Defines the gating mechanism, which selectively activates or deactivates connections based on triggers.
- **GatedLinear class**: A PyTorch layer modified to accommodate dynamic gating. This layer can adjust connections based on gate states, allowing the network to adapt its pathways on-the-fly.

## Running the Notebook

### In Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hannes-sistemica/Gating-first-implementation/blob/main/Gated_Network_Model_Simple_Demo.ipynb)

### With Dev Container Locally

To run the Gating model notebook, follow these steps:

#### Prerequisites

Ensure you have:
- **Docker**: To run the development container.
- **Visual Studio Code (VS Code)** with the **Remote - Containers** extension.

#### Steps to Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/gating.git
   cd gating