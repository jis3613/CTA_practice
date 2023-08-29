# Continual Test-time Adaptation Practice code

## Acknowledgement

This repository is modified version of official implemnetation of the paper: "MECTA: Memory-Economic Continual Test-time Adaptation." Junyuan Hong, Lingjuan Lyu, Jiayu Zhou, and Michael Spranger. ICLR 2023.
All the credits for developing this code goes to the authors of "MECTA: Memory-Economic Continual Test-time Adaptation.".
[paper](https://openreview.net/forum?id=N92hjSf5NNh) / [code](https://github.com/SonyAI/MECTA)
###### Copyright 2023, Sony AI, Sony Corporation of America, All rights reserved.

This repository is created to practice continual test-time adaptation code for educational use only.

## Getting Started

### Installation

1. Install packages.
    ```shell
    conda create --name mecta python=3.7
    conda activate mecta
    pip install -r requirements.txt  # work with cuda=10.2
    # NOTE if you work with cuda=11.3, do this to update otherwise not working.
    # cuda=11.3,11.4
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 -U
    pip install git+https://github.com/RobustBench/robustbench.git
    ```
2. Modify `data_root` in `utils/config.py` pointing to the data root.

### Quick Example

Run the bash script file `train_cifar100c.sh` in scripts for contual test-time adaptation with different methods.
```shell
cd ..

### Tent ###

CUDA_VISIBLE_DEVICES=1 python cta_eval.py \
    --data=cifar100 --alg=tent --model=rb_ResNeXt29_32x4d --batch_size=64 --lr=0.0001 \
    --exp_name lr_0.0002_b64 \
    --seed 2020

### EATA ###

CUDA_VISIBLE_DEVICES=1 python cta_eval.py \
    --data=cifar100 --alg=eata --model=rb_ResNeXt29_32x4d --batch_size=64 --lr=0.00025 \
    --exp_name lr_0.00025_b64 \
    --seed 2020

### Cotta ###

CUDA_VISIBLE_DEVICES=1 python cta_eval.py \
    --data=cifar100 --alg=cotta --model=rb_ResNeXt29_32x4d --batch_size=64 --lr=0.001 \
    --exp_name lr_0.001_b64 \
    --seed 2020
```

## Project Structure

* `cta_eval.py`: Main entry for running experiments.
* `algorithm`: Implementations of CTA algorithms.
