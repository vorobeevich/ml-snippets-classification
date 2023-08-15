[![DOI](https://zenodo.org/badge/678971229.svg)](https://zenodo.org/badge/latestdoi/678971229)
# Machine learning code snippets semantic classification
This repository contains the source code of experiments from the paper **"Machine learning code snippets semantic classification"** (*Valeriy Berezovskiy*, *Anastasia Gorodilova*, *Ekaterina Trofimova*, *Andrey Ustyuzhanin*).

# Preparation
Start by cloning the repository: 

```git clone https://github.com/vorobeevich/ml-snippets-classification```

We **highly** recommend using conda for experiments: [Anaconda](https://www.anaconda.com/download).

After installation, make a new environment:

```conda create --name cssc```

```conda activate cssc```

Install the libraries from the **requirements.txt**. Torch versions may differ depending on your GPU: [Start Locally | PyTorch](https://pytorch.org/get-started/locally/)

# Data 
Download the marked up data (**7947 snippets**), as well as the result of the partition algorithm from our Google Drive:

```chmod 777 /src/scripts/load_data.sh```

```./src/scripts/load_data.sh```

You can download the full version of **Code4ML** dataset (marked up data, a total set of 2.5 million snippets, our model predictions on all data) on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7733823.svg)](https://doi.org/10.5281/zenodo.7733823)

Also, you can read the paper about **Code4ML** Dataset: [Code4ML: a Large-scale Dataset of annotated Machine Learning Code](https://arxiv.org/abs/2210.16018).
# Usage
To reproduce any experiment from our article, it is enough to run the training script with the desired config.
Note that the result is non-deterministic (even with a fixed random seed) on various platforms due to the nature of libraries such as torch.

```python src/scripts/train.py --device [ID OF CUDA DEVICE] --config src/configs/[CHOOSE CONFIG TO RUN]```
