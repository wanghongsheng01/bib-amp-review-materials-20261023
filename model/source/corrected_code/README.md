# De novo Antimicrobial Peptides with Specific Functions Using Two-stage Multimodal Deep Probabilistic Generative Model

## Menu
- [Introduction](#introduction)
- [Installation](#installation)
- [Data for training](#data-for-training)
- [Usage](#Usage)


## Introduction

This project is used to train a two-stages antimicrobial peptide generation model, 
which can control the targeted generation of antimicrobial peptides with specific functions based on 
the labels of the second-stage training data.
![img.png](img.png)
## Installation

Please ensure that Conda is installed, then use the terminal to execute the following commands to install the environment 
dependencies required for this project:

```bash
# conda create env
conda env create -f amp_env.yml
# activate env
conda activate AMP
```

## data for training
We need to process the data for the first stage and the second stage separately. 
In the `datasets/` directory, we have provided example files of the dataset format 
(just given as format example, the complete dataset can 
be obtained from [Google Drive](https://drive.google.com/drive/folders/1SUp90gpuagtUUZmM3s6sXRqIRRvsZfcz?usp=sharing), and you can also refer to the `create_datasets.sh` script 
we provided to modify the dataset according to your needs).


## Usage
The training is divided into two stages. Please follow the steps given below and execute the corresponding 
commands in the order of step1 and step2.

### Step1:
Training of the multimodal flow-guided VAE. 
Ensure that `datasets_vae_train.csv` and `datasets_vae_val.csv` are placed in the `datasets/` directory, 
and synchronize these two paths to `binary_data_dir:` and `val_data_dir:` parameters in `configs/train_vae.yaml`.

Begin VAE training with:
```bash
bash train_vae_flow.sh
```
The relevant parameters for VAE training can be adjusted in `configs/train_vae.yaml`.
After the first stage of training, the model parameters will be saved in the `checkpoints` folder. 
Place the desired `.ckpt` type network weight file in the root directory and rename it to `VAE_flow.ckpt` 
to be used in the second stage of training.

### Step2:
Training of the conditionally controlled diffusion model.
Ensure that `datasets_for_diff.csv` is placed in the `datasets/` directory, and synchronize this 
path to `config` in `train_diffusion.py`.

Set the `mode` parameter of `config` in `train_diffusion.py` to 'train', then execute
```bash
python diffuison.py
```
to start training. The training hyperparameters can be modified in `config` of `train_diffusion.py`.
After training, set the `mode` parameter of `config` in `train_diffusion.py` to 'infer' and execute
```bash
python diffuison.py
```
again to generate new AMPs of given classes.