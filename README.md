# ADversarial Multi-Dimentional Molecular Graph Contrastive Learning (AD-MDMGCL)

## Prerequisites

- [Anaconda](https://docs.anaconda.com/anaconda/install/index.html)

## Setup

```bash
git clone https://github.com/kiibo382/AD-MDMGCL
conda install -c conda-forge mamba
!mamba env create -f environment.yml > /dev/null 2>&1
conda activate AD-MDMGCL
```

## Pre-train a model

```bash
python train.py --config=configs/pre-train_QM9.yml
```

You can monitor the pre-training process on tensorboard.

```bash
tensorboard --logdir=runs --port=6006
```

## Fine-tune a model

```bash
python train.py --config=configs/tune_QM9_homo.yml
```


## Acknowledgements

- [3DInfomax](https://github.com/HannesStark/3DInfomax)
- [AD-GCL](https://github.com/susheels/adgcl)
