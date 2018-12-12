#!/usr/bin/env python

from model import SentenceVAE, Actor
from utils import to_var, idx2word, interpolate, preprocess_nt, PHRASE_TAGS
import argparse
import json
import numpy as np
import os
import pickle
import torch
from multiprocessing import Pool
import time

# load parser
with open('parsers/viterbi_parser.pkl', 'rb') as f:
    PARSER = pickle.loads(f.read())

# labels for conditional generation
LABELS = [
    [0,0,0,0,0,0],
    [1,0,0,0,0,0],
    [0,1,0,0,0,0],
    [0,0,1,0,0,0],
    [0,0,0,1,0,0],
    [0,0,0,0,1,0],
    [0,0,0,0,0,1],
]

LABEL_NAMES = ['NONE']
LABEL_NAMES.extend(PHRASE_TAGS)


def pickle_it(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return(pickle.loads(f.read()))


def get_parse(sentence, n_elems=15):
    """parses the input sentence and returns the parse tree"""
    sentence = sentence.split()
    if len(sentence) > n_elems:
        sentence = sentence[:n_elems]
    else:
        sentence = sentence[:-1]

    try:
        output = list(PARSER.parse(sentence))
    except:
        return(['null'])

    # TODO: unclear if output=False is possible
    if output:
        return(output[0])
    else:
        return(['null'])


def get_parses(sentences):
    """runs viterbu in parallel across all sentences submitted"""
    pool = Pool() # required for multicore
    try:
        t1 = time.time()
        parses = pool.map_async(get_parse, sentences).get(9999999)
        pool.close()
        t2 = time.time()
        print('preprocessed all sentences in {} min'.format((t2-t1)/60.0))
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        sys.exit(1)

    return(parses)


def get_productions(productions):
    """get a preprocessed, string format of the nonterminals in the tree"""
    tags = []
    for production in productions:
        tags.append(preprocess_nt(production._lhs).unicode_repr())

    return(tags)


def find_tags_in_parse(phrase_tags, parses):
    tags = np.zeros((len(parses), len(phrase_tags)))
    for i, parse in enumerate(parses):
        try:
            productions = get_productions(parse.productions())
        except:
            productions = ['null']

        for j, tag in enumerate(PHRASE_TAGS):
            if tag in productions:
                tags[i, j] = 1

    return(tags)


def remove_bad_samples(samples, pct_unk=0.5):
    """
    first, remove trailing unks. after that, removes sentences that are over
    pct_unk% <UNK>s.
    """
    output_samples = []
    for sample in samples:
        tmp = sample.split()[:-1] # removes <eos>

        # remove trailing <unk>s
        unk_idx = np.where(np.array(tmp) == '<unk>')[0]
        last_idx = len(tmp)-1

        # if last entry is an <unk>, search backwards for the end of the chain
        # but only do this if the sentence is at least two <unk>s
        if last_idx in unk_idx and last_idx >= 2:
            while tmp[last_idx] == tmp[last_idx-1]:
                last_idx -= 1

                if last_idx == 1:
                    break

            tmp = tmp[:last_idx+1]

        n_unk = len(np.where(np.array(tmp) == '<unk>')[0])
        n_all = len(tmp)

        if n_unk/float(n_all) <= pct_unk:
            output_samples.append(' '.join(tmp))

    return(output_samples)


def get_sents_and_tags(samples, i2w, w2i):
    """
    preprocesses sentences, gets parses, and then returns phrase_tag occourance
    """
    samples = idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>'])
    samples = remove_bad_samples(samples)
    parses = get_parses(samples)
    tags = find_tags_in_parse(PHRASE_TAGS, parses)

    return(samples, tags)


def main(args):

    with open(args.data_dir+'/ptb.vocab.json', 'r') as file:
        vocab = json.load(file)

    # required to map between integer-value sentences and real sentences
    w2i, i2w = vocab['w2i'], vocab['i2w']

    # make sure our models for the VAE and Actor exist
    if not os.path.exists(args.load_vae):
        raise FileNotFoundError(args.load_vae)

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
    model.eval()
    print("vae model loaded from %s"%(args.load_vae))

    # to run in constraint mode, we need the trained generator
    if args.constraint_mode:
        if not os.path.exists(args.load_actor):
            raise FileNotFoundError(args.load_actor)

        actor = Actor(
            dim_z=args.latent_size, dim_model=2048, num_labels=args.n_tags)
        actor.load_state_dict(
            torch.load(args.load_actor, map_location=lambda storage, loc:storage))
        actor.eval()
        print("actor model loaded from %s"%(args.load_actor))

    if torch.cuda.is_available():
        model = model.cuda()
        if args.constraint_mode:
            actor = actor.cuda() # TODO: to(self.devices)

    if args.sample:
        print('*** SAMPLE Z: ***')
        # get samples from the prior
        sample_sents, z = model.inference(n=args.num_samples)
        sample_sents, sample_tags = get_sents_and_tags(sample_sents, i2w, w2i)
        pickle_it(z.cpu().numpy(), 'samples/z_sample_n{}.pkl'.format(args.num_samples))
        pickle_it(sample_sents, 'samples/sents_sample_n{}.pkl'.format(args.num_samples))
        pickle_it(sample_tags, 'samples/tags_sample_n{}.pkl'.format(args.num_samples))
        print(sample_sents, sep='\n')

        if args.constraint_mode:

            print('*** SAMPLE Z_PRIME: ***')
            # get samples from the prior, conditioned via the actor
            all_tags_sample_prime = []
            all_sents_sample_prime = {}
            all_z_sample_prime = {}
            for i, condition in enumerate(LABELS):

                # binary vector denoting each of the PHRASE_TAGS
                labels = torch.Tensor(condition).repeat(args.num_samples, 1).cuda()

                # take z and manipulate using the actor to generate z_prime
                z_prime = actor.forward(z, labels)

                sample_sents_prime, z_prime = model.inference(
                    z=z_prime, n=args.num_samples)
                sample_sents_prime, sample_tags_prime = get_sents_and_tags(
                    sample_sents_prime, i2w, w2i)
                print('conditoned on: {}'.format(condition))
                print(sample_sents_prime, sep='\n')
                all_tags_sample_prime.append(sample_tags_prime)
                all_sents_sample_prime[LABEL_NAMES[i]] = sample_sents_prime
                all_z_sample_prime[LABEL_NAMES[i]] = z_prime.data.cpu().numpy()
            pickle_it(all_tags_sample_prime, 'samples/tags_sample_prime_n{}.pkl'.format(args.num_samples))
            pickle_it(all_sents_sample_prime, 'samples/sents_sample_prime_n{}.pkl'.format(args.num_samples))
            pickle_it(all_z_sample_prime, 'samples/z_sample_prime_n{}.pkl'.format(args.num_samples))

    if args.interpolate:
        # get random samples from the latent space
        z1 = torch.randn([args.latent_size]).numpy()
        z2 = torch.randn([args.latent_size]).numpy()
        z = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=args.num_samples-2)).float())

        print('*** INTERP Z: ***')
        interp_sents, _ = model.inference(z=z)
        interp_sents, interp_tags = get_sents_and_tags(interp_sents, i2w, w2i)
        pickle_it(z.cpu().numpy(), 'samples/z_interp_n{}.pkl'.format(args.num_samples))
        pickle_it(interp_sents, 'samples/sents_interp_n{}.pkl'.format(args.num_samples))
        pickle_it(interp_tags, 'samples/tags_interp_n{}.pkl'.format(args.num_samples))
        print(interp_sents, sep='\n')

        if args.constraint_mode:
            print('*** INTERP Z_PRIME: ***')
            all_tags_interp_prime = []
            all_sents_interp_prime = {}
            all_z_interp_prime = {}

            for i, condition in enumerate(LABELS):

                # binary vector denoting each of the PHRASE_TAGS
                labels = torch.Tensor(condition).repeat(args.num_samples, 1).cuda()

                # z prime conditioned on this particular binary variable
                z_prime = actor.forward(z, labels)

                interp_sents_prime, z_prime = model.inference(
                    z=z_prime, n=args.num_samples)
                interp_sents_prime, interp_tags_prime = get_sents_and_tags(
                    interp_sents_prime, i2w, w2i)
                print('conditoned on: {}'.format(condition))
                print(interp_sents_prime, sep='\n')
                all_tags_interp_prime.append(interp_tags_prime)
                all_sents_interp_prime[LABEL_NAMES[i]] = interp_sents_prime
                all_z_interp_prime[LABEL_NAMES[i]] = z_prime.data.cpu().numpy()

            pickle_it(all_tags_interp_prime, 'samples/tags_interp_prime_n{}.pkl'.format(args.num_samples))
            pickle_it(all_sents_interp_prime, 'samples/sents_interp_prime_n{}.pkl'.format(args.num_samples))
            pickle_it(all_z_interp_prime, 'samples/z_interp_prime_n{}.pkl'.format(args.num_samples))

    import IPython; IPython.embed()


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

    # outputs
    parser.add_argument('-s', '--sample', action='store_true')
    parser.add_argument('-i', '--interpolate', action='store_true')

    args = parser.parse_args()
    args.rnn_type = args.rnn_type.lower()
    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert 0 <= args.word_dropout <= 1

    main(args)

