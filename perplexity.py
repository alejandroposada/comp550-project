#!/usr/bin/env python

from collections import OrderedDict
from ptb import PTB
import argparse
import numpy as np
import collections

SMOOTH_VAL = 1 # if 1, skip unknown words, if 0 < SMOOTH_VAL < 1, assign prob

def perplexity(testset, model):
    """computes perplexity of the unigram model on a set of sentences"""

    entropy = 0
    total_skipped = 0

    N = sum([len(sentence.split()) for sentence in testset])

    for sentence in testset:
        this_sentence_prob, n_skipped = sentence_prob(sentence, model)
        entropy += np.log2(this_sentence_prob)
        total_skipped += n_skipped # if SMOOTH_VAL == 1

    N -= total_skipped # adjust N for the number of skipped words
    perplexity = 2 ** (-1 / N * entropy)

    return(perplexity)


def sentence_prob(sentence, model):
    """get the probability of a sentence"""
    prob = 1
    skipped = 0

    for word in sentence.split():
        this_prob = model[word]

        # when SMOOTH_VAL = 1, we decide not to count unseen words entirely.
        # so we reduce N, in order to later divide by the correct
        # number of words.

        if this_prob == 1:
            skipped += 1

        prob *= this_prob

    return(prob, skipped)


def unigram(tokens):
    """constructs the unigram language model"""

    # don't count these tokens, they are not supposed to be in the language
    stops = ['<', '>', '*-4', '\\*', '\\*\\*']

    # lambda defines a default value for words encountered that are not in
    # this corpus
    model = collections.defaultdict(lambda: SMOOTH_VAL)

    for f in tokens:

        if f in stops:
            continue

        try:
            model[f] += 1
        except KeyError:
            model[f] = 1
            continue

    # normalize by total count to get probabilities
    total = sum(model.values())
    for word in model:
        model[word] = model[word] / total

    return(model)


def load_prime_samples(path):
    d = np.load(path)
    samples = []
    for k, v in d.items():
        samples += v
    return samples


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

    # load samples
    samples = np.load('samples/sents_sample_n250.pkl')
    samples_prime = load_prime_samples('samples/sents_sample_prime_n250.pkl')
    samples_prime_dict = np.load('samples/sents_sample_prime_n250.pkl')

    f = open('results/perplexity.csv', 'w')
    f.write('data_type,sample_type,perplexity\n')

    for data_type in ['train', 'valid']:
        datasets = OrderedDict()
        datasets[data_type] = PTB(
            data_dir=args.data_dir,
            split=data_type,
            create_data=False,
            max_sequence_length=args.max_sequence_length,
            min_occ=args.min_occ
        )

        corpus = make_corpus(datasets[data_type])
        model = unigram(corpus)

        # compute and save perplexities
        f.write('{},all_z_prime,{:.2f}\n'.format(
            data_type, perplexity(samples_prime, model)))
        for k in samples_prime_dict.keys():
            f.write('{},{}_z_prime,{}\n'.format(
                data_type, k, perplexity(samples_prime_dict[k], model)))
        f.write('{},z,{:.2f}\n'.format(data_type, perplexity(samples, model)))

    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--max_sequence_length', type=int, default=60)
    parser.add_argument('--min_occ', type=int, default=1)
    args = parser.parse_args()

    main(args)

