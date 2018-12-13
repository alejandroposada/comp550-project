#!/bin/bash

# trains the VAE
./train_vae.py -bin /home/jdv/pytorch -ep 100 -tb -wd 0.25

# trains the conditional-generator actor critic pair
#./train_ac.py --vae_path save_model/E99.pytorch -wd 0.25

# generates outputs for evaluation of model
#./inference.py --load_vae save_model/E99.pytorch --load_actor save_model/actor_model99.path.tar -n 250 --constraint_mode --sample --interpolate

