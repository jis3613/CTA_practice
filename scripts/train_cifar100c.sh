#!/usr/bin/env bash

cd ..

#### Tent ###

CUDA_VISIBLE_DEVICES=1 python cta_eval.py \
    --data=cifar100 --alg=tent --model=rb_ResNeXt29_32x4d --batch_size=64 --lr=0.0001 \
    --exp_name lr_0.0001_b64 \
    --seed 2020

#### EATA ###

CUDA_VISIBLE_DEVICES=1 python cta_eval.py \
    --data=cifar100 --alg=eata --model=rb_ResNeXt29_32x4d --batch_size=64 --lr=0.00025 \
    --exp_name lr_0.00025_b64 \
    --seed 2020

#### Cotta ###

CUDA_VISIBLE_DEVICES=1 python cta_eval.py \
    --data=cifar100 --alg=cotta --model=rb_ResNeXt29_32x4d --batch_size=64 --lr=0.001 \
    --exp_name lr_0.001_b64 \
    --seed 2020