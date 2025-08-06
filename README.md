# LLM from Scratch: Educational Transformer Implementation

This repository provides a detailed, modular implementation of a Large Language Model (LLM) from scratch, inspired by architectures like GPT-2. The code is designed for students and researchers to understand the mathematics, dimensionality, and engineering behind transformer-based models. Each module is explained in depth, with references to the underlying theory and practical considerations.

---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Getting Started](#getting-started)
4. [Usage](#usage)
5. [Detailed Concepts & Mathematics](#detailed-concepts--mathematics)

   * [Feedforward Layer](#feedforward-layer)
   * [GeLU Activation](#gelu-activation)
   * [Layer Normalization](#layer-normalization)
   * [Positional Encoding](#positional-encoding)
   * [Token Embeddings](#token-embeddings)
   * [Attention Mechanisms](#attention-mechanisms)
   * [Shortcut Connections](#shortcut-connections)
   * [Tokenizer](#tokenizer)
   * [Data Loader](#data-loader)
6. [Customization](#customization)
7. [License](#license)
8. [Acknowledgements](#acknowledgements)

---

## Features

* Modular transformer blocks with clear separation of concerns
* Multiple attention mechanisms (self, causal, multi-head, Bahdanau)
* Custom tokenizer and embedding layers
* Layer normalization and GeLU activation
* Data loader for text datasets
* Example datasets (`hamlet.txt`, `the_verdict.txt`)
* PyTorch-based for extensibility and GPU support

## Project Structure

```
pyproject.toml           # Project dependencies and configuration
concepts/                # Core concepts and layer implementations
    feedforward_layer.py         # Fully connected layers and math
    GeLU.py                     # Gaussian Error Linear Unit activation
    io_pairs.py                 # Input-output pair handling
    layer_norm.py               # Layer normalization math
    positional_encoding.py      # Sinusoidal positional encoding
    short_cut_connection.py     # Residual/shortcut connections
    token_embeddings.py         # Token embedding logic
    tokenizer.py                # Tokenizer implementation
attention/               # Attention mechanisms
    badhanau_attention.py       # Bahdanau (additive) attention
    causal_attention.py         # Causal (masked) attention
    multi_head_attention.py     # Multi-head attention math
    self_attention.py           # Self-attention implementation
    weight_split_multihead_attention.py # Weight splitting for multi-head
custom_llm/              # Model and test scripts
    model.py                    # Full model definition
    model_test.py               # Model testing and evaluation
    trained_model.pt            # Saved model weights
    components/                 # Submodules for model architecture
        dataloader.py           # Data loading utilities
        embedding_layer.py      # Embedding layer logic
        feedforward_network.py  # Feedforward network math
        gelu.py                 # GeLU activation
        layer_normalizing.py    # Layer normalization
        multi_head_attention.py # Multi-head attention
        positional_emdedding.py # Positional embedding
        tokenizer.py            # Tokenizer
        transformer_block.py    # Transformer block logic
data/                    # Sample datasets
    hamlet.txt
    the_verdict.txt
```

## Getting Started

### Prerequisites

* Python 3.8+
* [PyTorch](https://pytorch.org/)

Install dependencies:

```bash
pip install -r requirements.txt
```

or, if using `pyproject.toml`:

```bash
pip install .
```

## Usage

To train the model, run:

```bash
cd custom_llm
uv run model.py
```

To test the model, you can use:

```bash
uv run model_test.py
```

You can modify `model.py` or modify the components in `custom_llm/components` to experiment with different model configurations and components.

## Detailed Concepts & Mathematics

### Feedforward Layer (`concepts/feedforward_layer.py`, `components/feedforward_network.py`)

Implements the fully connected layers in the transformer block. The feedforward network typically consists of two linear transformations with a non-linearity (GeLU) in between:

```math
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
```

* **Input Dimensionality:** `(batch_size, seq_len, d_model)`
* **Hidden Layer:** `(batch_size, seq_len, d_ff)`
* **Output:** `(batch_size, seq_len, d_model)`

### GeLU Activation (`concepts/GeLU.py`, `components/gelu.py`)

Implements the Gaussian Error Linear Unit:

```math
\text{GeLU}(x) = x \cdot \Phi(x)
```

where $\Phi(x)$ is the cumulative distribution function of the standard normal distribution. GeLU is smoother than ReLU and helps with gradient flow.

### Layer Normalization (`concepts/layer_norm.py`, `components/layer_normalizing.py`)

Normalizes the inputs across the features for each token:

```math
\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta
```

where $\mu$ and $\sigma$ are the mean and standard deviation across the last dimension, and $\gamma$, $\beta$ are learnable parameters.

### Positional Encoding (`concepts/positional_encoding.py`, `components/positional_emdedding.py`)

Adds information about token position using sinusoidal functions:

```math
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
```

```math
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
```

* **Input:** `(seq_len, d_model)`
* **Output:** `(seq_len, d_model)`

### Token Embeddings (`concepts/token_embeddings.py`, `components/embedding_layer.py`)

Maps token indices to dense vectors:

```math
\text{Embedding}(x) = W[x]
```

where $W$ is the embedding matrix of shape `(vocab_size, d_model)`.

### Attention Mechanisms (`attention/`, `components/multi_head_attention.py`, `components/transformer_block.py`)

#### Self-Attention (`attention/self_attention.py`)

Calculates attention scores for each token with respect to all others:

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

* **Q, K, V Dimensionality:** `(batch_size, seq_len, d_model)`
* **Output:** `(batch_size, seq_len, d_model)`

#### Multi-Head Attention (`attention/multi_head_attention.py`, `components/multi_head_attention.py`)

Splits the model into multiple heads to learn different representations:

```math
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
```

Each head computes its own attention.

* **Heads:** $h$
* **Each Head:** `(batch_size, seq_len, d_k)`
* **Output:** `(batch_size, seq_len, d_model)`

#### Causal Attention (`attention/causal_attention.py`)

Masks future tokens to prevent information leakage during training.

#### Bahdanau Attention (`attention/badhanau_attention.py`)

Implements additive attention, commonly used in sequence-to-sequence models.

#### Weight Split Multi-Head (`attention/weight_split_multihead_attention.py`)

Splits weights for each head for efficiency and clarity.

### Shortcut Connections (`concepts/short_cut_connection.py`)

Implements residual connections:

```math
\text{Output} = \text{Layer}(x) + x
```

Helps with gradient flow and stabilizes training.

### Tokenizer (`concepts/tokenizer.py`, `components/tokenizer.py`)

Converts raw text into token indices. Can be customized for different tokenization strategies (word, subword, character).

### Data Loader (`components/dataloader.py`)

Loads and batches text data for training. Handles padding, batching, and shuffling.

---

## Customization

* Add new layers or attention mechanisms in the `concepts/` or `attention/` folders.
* Use your own text data by placing files in the `data/` directory and updating the data loader.
* Adjust hyperparameters and model architecture in `main.py` and `custom_llm/model.py`.

---

## License

This project is for educational purposes. See LICENSE for details.

## Acknowledgements

* Inspired by OpenAI GPT-2 and transformer architectures.
* Thanks to the PyTorch community for open-source tools and documentation.

---

For questions, suggestions, or contributions, feel free to open an issue or pull request!
