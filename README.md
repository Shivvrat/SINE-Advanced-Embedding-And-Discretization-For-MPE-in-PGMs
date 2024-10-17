# SINE: Scalable MPE Inference for Probabilistic Graphical Models using Advanced Neural Embeddings

**Abstract:**

> Our paper builds on the recent trend of using neural networks trained with self-supervised or supervised learning to solve the Most Probable Explanation (MPE) task in discrete graphical models. At inference time, these networks take an evidence assignment as input and generate the most likely assignment for the remaining variables via a single forward pass. We address two key limitations of existing approaches: (1) the inability to fully exploit the graphical model's structure and parameters, and (2) the suboptimal discretization of continuous neural network outputs. Our approach embeds model structure and parameters into a more expressive feature representation, significantly improving performance. Existing methods rely on standard thresholding, which often yields suboptimal results due to the non-convexity of the loss function. We introduce two methods to overcome discretization challenges: (1) an external oracle-based approach that infers uncertain variables using additional evidence from confidently predicted ones, and (2) a technique that identifies and selects the highest-scoring discrete solutions near the continuous output. Experimental results on various probabilistic models demonstrate the effectiveness and scalability of our approach, highlighting its practical impact.

## Table of Contents

- [SINE: Scalable MPE Inference for Probabilistic Graphical Models using Advanced Neural Embeddings](#sine-scalable-mpe-inference-for-probabilistic-graphical-models-using-advanced-neural-embeddings)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Probabilistic Graphical Models:](#probabilistic-graphical-models)
  - [Main Components](#main-components)
  - [Key Features](#key-features)
  - [Usage](#usage)
    - [Important Arguments](#important-arguments)
    - [Example Command](#example-command)
    - [Embedding Types and Binarization Methods](#embedding-types-and-binarization-methods)
      - [For `embedding-type`:](#for-embedding-type)
      - [For `threshold_type`:](#for-threshold_type)
  - [Project Structure](#project-structure)
  - [Running Experiments](#running-experiments)
  - [Stored Outputs](#stored-outputs)
  - [Notes](#notes)

## Overview

The main entry point of the project is `main.py`, which sets up the experiment configuration, initializes the necessary components, and runs the training and testing processes.

## Installation

This project uses Conda for environment management. Follow these steps to set up the environment:

1. Create a new Conda environment using the provided `sine_mpe.yml` file:

   ```
   conda env create -f sine_mpe.yml
   ```
2. Activate the environment:

   ```
   conda activate mmap-advanced
   ```
3. If the yaml file does not work, install dependencies using pip:

   ```
   pip install -r requirements.txt
   ```

This will install all necessary dependencies, including PyTorch, CUDA support, and other required libraries.

**Note**: *Make sure you have CUDA-compatible hardware and drivers installed if you plan to use GPU acceleration.*

## Probabilistic Graphical Models:

We used PGM models from the uai benchmarks. Please download the models from the uai06 and uai22 benchmarks and place them in the the corresponding folders. These can be downloaded from:

1. https://sli.ics.uci.edu/~ihler/uai-data/index.html
2. https://uaicompetition.github.io/uci-2022/results/benchmarks/

**Note**: *These are cited/anonymous dataset repository links. The datasets are publicly available and can be downloaded from the respective links.*

## Main Components

1. **ConfigManager**: Handles configuration settings and initializes the experiment environment.
2. **DataManager**: Manages data loading and preprocessing.
3. **ModelManager**: Initializes and manages the neural network models.
4. **Trainer**: Orchestrates the training and testing processes.

## Key Features

- Support for various model types
  - NN,
  - Transformer
- Support for various embedding types
  - embedding based on query only (discrete),
  - embedding based on query and the structure and parameters of the pgm (hgnn)
- Support for various binarization methods
  - thresholding (basic),
  - oracle based (branch and bound),
  - selecting the highest scoring discrete solution near the continuous output (knearest_binary_vectors)
- Flexible configuration options for experiments
- Comprehensive logging and experiment tracking
- Storage of outputs in the specified experiment directory

## Usage

To run an experiment, use the following command inside the `mpe_advanced_models/model_1` folder:

```
python src/main.py [arguments]
```

All the scripts are in the `mpe_advanced_models/model_1/src` folder.

### Important Arguments

Some key arguments include:

- `--model`: Choose between "nn" or "transformer" models
- `--pgm`: Select the type of Probabilistic Graphical Model
- `--dataset`: Specify the dataset/benchmark to use
- `--embedding-type`: Choose the embedding type for inputs
- `--binarization-method`: Choose the binarization method for outputs

### Example Command

```bash
    python3 src/main.py \
        --no-debug \
        --query-prob=<query_prob> \
        --model=<model> \
        --dataset <dataset_name> \
        --task <task> \
        --partition_type <partition_type> \
        --embedding-type <embedding_type> \
        --experiment-dir <experiment_dir> \
        --pgm <dataset_type> \
        --threshold_type <binarization_method>
```

### Embedding Types and Binarization Methods

#### For `embedding-type`:

```94:99:mpe_advanced_models/model_1/src/utils_folder/arguments.py
    parser.add_argument(
        "--embedding-type",
        default="hgnn",
        choices=["discrete", "hgnn",],
        help="Embedding type (default: discrete).",
    )
```

- `discrete` (baseline):

  - Uses a simple discrete embedding for the input variables
  - Only uses the information of the query and evidence to predict the missing variable
- `hgnn` (Hypergraph Neural Network):

  - Utilizes a hypergraph structure to capture complex relationships between variables
  - Captures the structure and parameters of the probabilistic graphical model more effectively

#### For `threshold_type`:

```412:424:mpe_advanced_models/model_1/src/utils_folder/arguments.py
    parser.add_argument(
        "--threshold_type",
        type=str,
        default="basic",
        choices=[
            "basic",
            "knearest_binary_vectors",
            "branch_and_bound",
            "knearest_binary_vectors,branch_and_bound",
        ],
        help="Threshold type (default: basic).",
    )
```

- `basic` (baseline):

  - Simple thresholding method
  - Applies a fixed threshold (default 0.50) to convert continuous outputs to binary
- `knearest_binary_vectors`:

  - Finds the nearest binary vectors to the continuous output
  - Can improve accuracy by considering multiple possible solutions
- `branch_and_bound`:

  - Uses an oracle based approach for thresholding
  - Can find optimal or near-optimal solutions by considering additional evidence
- `knearest_binary_vectors,branch_and_bound`:

  - Combines the k-nearest binary vectors approach with the oracle based approach
  - Potentially offers the benefits of both methods for improved accuracy

These embedding and thresholding methods correspond to different approaches proposed in the paper, allowing users to experiment with various techniques for solving the Most Probable Explanation (MPE) task in probabilistic graphical models.

## Project Structure

- `main.py`: Entry point of the project
- `utils_folder/arguments.py`: Defines and processes command-line arguments
- `util_classes/`:
  - `config.py`: Manages configuration settings
  - `data_manager.py`: Handles data loading and processing
  - `model_manager.py`: Manages model initialization and optimization
- `trainer.py`: Implements the training and testing logic
- `nn_scripts.py`: Contains neural network-related functions
- `test_and_eval.py`: Implements testing and evaluation procedures

## Running Experiments

1. Set up your environment and install dependencies
2. Configure your experiment by selecting appropriate command-line arguments
3. Run `src/main.py` with the desired arguments
4. Monitor the output for logging information and results

## Stored Outputs

- The script generates logs, performance metrics, and model checkpoints, all stored in the specified experiment directory.

## Notes

- The project uses PyTorch and Lightning for efficient model training
- Experiment results are saved and can be analyzed post-run
- Various thresholding techniques are available for output binarization

For more detailed information about specific components or functions, please refer to the respective source files in the project.
