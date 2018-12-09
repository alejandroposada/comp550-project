#!/bin/bash

./inference.py \
    --load_vae model/E99.pytorch \
    --load_actor save_model/actor_model99.path.tar \
    -n 5 \
    --constraint_mode

