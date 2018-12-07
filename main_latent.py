#!/usr/bin/env python

import argparse
from model import SentenceVAE
from ptb import PTB
from actor_critic import Actor, Critic
from ac_trainer import AC_Trainer
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
from multiprocessing import cpu_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_sequence_length', type=int, default=60)
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('--test', action='store_true')

    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)

    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--vae_path', type=str, default=None)   # TODO:default path

    args = parser.parse_args()
    print(args)

    cuda = not args.no_cuda and torch.cuda.is_available()

    # Get DataLoaders
    datasets = OrderedDict()
    for split in ['train', 'valid']:
        datasets[split] = PTB(
                              data_dir=args.data_dir,
                              split=split,
                              create_data=args.create_data,
                              max_sequence_length=args.max_sequence_length,
                              min_occ=args.min_occ
                              )
    trainDataLoader = DataLoader(
                                dataset=datasets['train'],
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=cpu_count(),
                                pin_memory=torch.cuda.is_available()
                                )

    validDataLoader = DataLoader(
                                dataset=datasets['valid'],
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=cpu_count(),
                                pin_memory=torch.cuda.is_available()
                                )

    num_tags = len(datasets['train'][0]['phrase_tags'])      # 5

    # Load trained VAE
    vae_model = SentenceVAE(
        vocab_size=datasets['train'].vocab_size,
        sos_idx=datasets['train'].sos_idx,
        eos_idx=datasets['train'].eos_idx,
        pad_idx=datasets['train'].pad_idx,
        unk_idx=datasets['train'].unk_idx,
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
        )
    checkpoint = torch.load(args.vae_path)
    vae_model.load_state_dict(checkpoint)
    vae_model.cuda() # TODO to.(self.device)

    actor = Actor(dim_z=args.latent_size,
                  dim_model=2048,
                  num_labels=num_tags)
    actor.cuda() # TODO: to(self.device)

    real_critic = Critic(dim_z=args.latent_size,
                         dim_model=2048,
                         num_labels=num_tags,
                         conditional_version=True)
    real_critic.cuda()

    attr_critic = Critic(dim_z=args.latent_size,
                         dim_model=2048,
                         num_labels=num_tags,
                         num_outputs=num_tags,
                         conditional_version=True)
    attr_critic.cuda()

    ac_trainer = AC_Trainer(vae_model=vae_model,
                            actor=actor,
                            real_critic=real_critic,
                            attr_critic=attr_critic,
                            num_epochs=args.epochs,
                            trainDataLoader=trainDataLoader,
                            valDataLoader=validDataLoader)

    # Train!
    print('\n Training has started \n')
    ac_trainer.train()
