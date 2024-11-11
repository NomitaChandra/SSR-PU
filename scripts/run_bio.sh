#!/bin/sh

python train_bio.py --data_dir ./dataset/chemdisgene \
    --transformer_type bert \
    --model_name_or_path ./pretrain/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --train_file train.json \
    --dev_file dev.json \
    --test_file test.json \
    --train_batch_size 8 \
    --test_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.06 \
    --num_train_epochs 30.0 \
    --seed 66 \
    --num_class 1 \
    --isrank 0 \
    --m_tag S-PU \
    --e 3.0
