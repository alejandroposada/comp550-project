#!/bin/bash

./inference.py \
    --load_vae model/E19.pytorch \
    --load_actor save_model/actor_model49.path.tar \
    --constraint_mode \
    -n 100

