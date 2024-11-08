#!/bin/sh

python train.py --data_dir ./dataset/docred \
    --transformer_type roberta \
    --model_name_or_path ../../pretrain/Roberta-large \
    --train_file train_annotated.json \
    --dev_file dev.json \
    --test_file test_revised.json \
    --train_batch_size 4 \
    --test_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-5 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.06 \
    --num_train_epochs 30.0 \
    --seed 66 \
    --num_class 96 \
    --isrank 0 \
    --m_tag S-PU \
    --e 3.0
