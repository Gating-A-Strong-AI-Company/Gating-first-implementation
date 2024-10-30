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

## Running the Notebooks

### In Colab

Below are links to open each notebook directly in Google Colab:

- **[Gated_Network_Model_Simple_Demo.ipynb](https://colab.research.google.com/github/hannes-sistemica/Gating-first-implementation/blob/main/Gated_Network_Model_Simple_Demo.ipynb)**: A simple demonstration of the gating mechanism.
- **[Gated_Network_Model_Simple_Comparison.ipynb](https://colab.research.google.com/github/hannes-sistemica/Gating-first-implementation/blob/main/Gated_Network_Model_Simple_Comparison.ipynb)**: Compares the gated network outputs with and without gating.
- **[Gated_Network_Model_Simple_2_Gating_Regions.ipynb](https://colab.research.google.com/github/hannes-sistemica/Gating-first-implementation/blob/main/Gated_Network_Model_Simple_2_Gating_Regions.ipynb)**: A model showcasing two gating regions.
- **[Gated_Network_Model_Simple_Learning_Process.ipynb](https://colab.research.google.com/github/hannes-sistemica/Gating-first-implementation/blob/main/Gated_Network_Model_Simple_Learning_Process.ipynb)**: Demonstrates a simple learning process with gating.

### Classic installation

See [official website](https://jupyter.org/install) for installation instructions to set up Jupyter notebooks locally.

### With Local Dev Container

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


## How to Contribute

We welcome contributions to improve and expand the Gating project! Follow these steps to get involved:

### Step 1: Fork the Repository
1. Go to the GitHub page of this repository and click on the "Fork" button at the top-right corner.
2. This will create a copy of the repository under your GitHub account.

### Step 2: Clone Your Fork
1. Clone the forked repository to your local machine:
   ```bash
   git clone https://github.com/<yourusername>/Gating-first-implementation.git
   cd gating
   ```

### Step 3: Create a New Branch
1. Before making any changes, create a new branch to keep your changes separate:
   ```bash
   git checkout -b your-branch-name
   ```

### Step 4: Make Changes and Test
1. Make your changes to the code or documentation.
2. If applicable, test your changes locally to ensure everything works as expected.

### Step 5: Commit and Push
1. Commit your changes with a descriptive commit message:
   ```bash
   git add .
   git commit -m "Description of changes made"
   ```
2. Push your changes to your fork:
   ```bash
   git push origin your-branch-name
   ```

### Step 6: Create a Pull Request
1. Go to the original repository on GitHub.
2. Click on the "Pull Requests" tab, then click "New Pull Request".
3. Select the branch you pushed to your fork, and open a pull request.
4. Add a descriptive title and summary of your changes, then submit the pull request.

Your contribution will be reviewed, and we’ll collaborate with you for any changes needed before merging. Thank you for helping make this project better!

### Additional Notes

- Please follow the coding standards and guidelines as closely as possible.
- Make sure to check the documentation and provide additional comments or explanations if your code changes are complex.
- If you are contributing a new feature or significant change, consider opening an issue first to discuss it with the maintainers.

Happy coding!