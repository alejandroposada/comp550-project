#!/usr/bin/env python

import os
import json
import torch
import argparse

from model import SentenceVAE
from actor_critic import Actor
from utils import to_var, idx2word, interpolate


def main(args):

    with open(args.data_dir+'/ptb.vocab.json', 'r') as file:
        vocab = json.load(file)

    w2i, i2w = vocab['w2i'], vocab['i2w']

    if not os.path.exists(args.load_vae):
        raise FileNotFoundError(args.load_vae)

    if not os.path.exists(args.load_actor):
        raise FileNotFoundError(args.load_actor)

    model = SentenceVAE(
        vocab_size=len(w2i),
        sos_idx=w2i['<sos>'],
        eos_idx=w2i['<eos>'],
        pad_idx=w2i['<pad>'],
        unk_idx=w2i['<unk>'],
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

    model.load_state_dict(
        torch.load(args.load_vae, map_location=lambda storage, loc: storage))
    print("vae model loaded from %s"%(args.load_vae))

    model.eval()

    if args.constraint_mode:
        actor = Actor(dim_z=args.latent_size,
                      dim_model=2048,
                      num_labels=args.n_tags)
        actor.load_state_dict(
            torch.load(args.load_actor, map_location=lambda storage, loc:storage))
        print("actor model loaded from %s"%(args.load_vae))

        actor.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        actor = actor.cuda() # TODO: to(self.devices)

    # get samples from the prior
    samples, z = model.inference(n=args.num_samples)

    # take z and manipulate them using the actor to generate z_prime
    if args.constraint_mode:
        labels = torch.Tensor([0,0,0,1,0,0]).repeat(args.num_samples, 1).cuda()
        z_prime = actor.forward(z, labels)
        samples_prime, z_prime = model.inference(z=z_prime, n=args.num_samples)

    print('*** SAMPLES Z: ***')
    print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

    if args.constraint_mode:
        print('*** SAMPLES Z_PRIME: ***')
        print(*idx2word(samples_prime, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

    z1 = torch.randn([args.latent_size]).numpy()
    z2 = torch.randn([args.latent_size]).numpy()
    z = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=args.num_samples-2)).float())
    samples, _ = model.inference(z=z)

    if args.constraint_mode:
        z_prime = actor.forward(z, labels)
        samples_prime, z_prime = model.inference(z=z_prime, n=args.num_samples)

    print('*** INTERP Z: ***')
    print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')
    if args.constraint_mode:
        print('*** INTERP Z_PRIME: ***')
        print(*idx2word(samples_prime, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_vae', type=str)
    parser.add_argument('-n', '--num_samples', type=int, default=10)

    parser.add_argument('-dd', '--data_dir', type=str, default='data')
    parser.add_argument('-ms', '--max_sequence_length', type=int, default=50)
    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')

    # conditional specific stuff
    parser.add_argument('-t',  '--n_tags', type=int, default=6)
    parser.add_argument('-cm', '--constraint_mode', action='store_true')
    parser.add_argument('-ca', '--load_actor', type=str)

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert 0 <= args.word_dropout <= 1

    main(args)

