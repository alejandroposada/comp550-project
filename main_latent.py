import argparse
from model import SentenceVAE
from ptb import PTB
from collections import OrderedDict
from actor_critic import Actor, Critic
from ac_trainer import AC_Trainer
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
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

    splits = ['train', 'valid'] + (['test'] if args.test else [])

    # Generates a data structure
    # TODO: load data
    datasets = OrderedDict()
    for split in splits:
        datasets[split] = PTB(
            data_dir=args.data_dir,
            split=split,
            create_data=args.create_data,
            max_sequence_length=args.max_sequence_length,
            min_occ=args.min_occ
        )

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

    # TODO: num_labels
    actor = Actor(dim_z=args.embedding_size,
                  dim_model=2048)
    real_critic = Critic(dim_z=args.embedding_size,
                         dim_model=2048,
                         conditional_version=True)
    attr_critic = Critic(dim_z=args.embedding_size,
                         dim_model=2048,
                         num_outputs=5,
                         conditional_version=True)              # TODO: num_outputs

    ac_trainer = AC_Trainer(vae_model=vae_model,
                            actor=actor,
                            real_critic=real_critic,
                            attr_critic=attr_critic,
                            num_epochs=args.num_epochs,
                            trainDataLoader=None,
                            valDataLoader=None)                 # TODO: dataloaders

    # Train!
    print('\n Training has started \n')
    ac_trainer.train()
