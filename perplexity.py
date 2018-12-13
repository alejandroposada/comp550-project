#!/usr/bin/env python

from utils import load_pickle
from collections import OrderedDict
from ptb import PTB
import argparse
import numpy as np
from nltk.probability import FreqDist


#def remove_pad(tokens):

def perplexity(testset, freqs):
    perplexity = 0
    N = float(freqs.N())

    for sentence in testset:
        perplexity += np.log2(sentence_prob(sentence, freqs))

    return(2**( (-1/N) * perplexity) )


def sentence_prob(sentence, freqs):
    prob = 1
    for word in sentence:
        prob *= freqs.freq(word)
    return(prob)


def make_corpus(dataset):
    corpus = []
    for i in range(len(dataset)):

        # 0 = <pad>
        idx = np.where(dataset[i]['input'] != 0)[0]

        # idx[1:] strips <sos> from beginning
        tokens = np.array(dataset[i]['input_str'])[idx[1:]]

        corpus.extend(tokens.tolist())

    return(corpus)


def main(args):

    datasets = OrderedDict()
    datasets['valid'] = PTB(
        data_dir=args.data_dir,
        split='valid',
        create_data=False,
        max_sequence_length=args.max_sequence_length,
        min_occ=args.min_occ
    )


    corpus = make_corpus(datasets['valid'])
    freqs = FreqDist(corpus)
    sents_sample = load_pickle('samples/sents_sample_n250.pkl')
    sents_sample_prime = load_pickle('samples/sents_sample_prime_n250.pkl')

    import IPython; IPython.embed()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--max_sequence_length', type=int, default=60)
    parser.add_argument('--min_occ', type=int, default=1)
    args = parser.parse_args()

    main(args)

