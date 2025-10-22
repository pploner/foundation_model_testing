<div align="center">

# Repository for loading and processing the COLLIDE-2V dataset and training for a foundation model

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

At its current state, this repository is loading data from the COLLIDE-2V dataset as saved on EOS, saving it in a vectorized form usable for training, and using it to train a toy tinyMLP classifier. The config allows for many setting with respect to selected features and processes from the dataset, and can be readily used to train different types of classifiers. Preprocessing is (apart from a basic (x-mean)/std normalization) not yet implemented and will be investigated as a next step.

The repository follows the lightning-hydra framework that is accessible here: <a href="https://github.com/ashleve/lightning-hydra-template"> . For more details on the structure of the code please consult the readme there.

Please send any feedback or suggestions to plonerp@ethz.ch.

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/pploner/foundation_model_testing.git
cd foundation_model_testing

# [OPTIONAL] create conda environment
conda create -n myenv python=3.10
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
