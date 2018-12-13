#!/usr/bin/env python

from collections import OrderedDict
from ptb import PTB
import argparse
import numpy as np
import collections


# computes perplexity of the unigram model on a test set
def perplexity(testset, model):
    perplexity = 0
    N = sum([len(sentence.split()) for sentence in testset])

    for sentence in testset:
        perplexity += np.log2(sentence_prob(sentence, model))
    return 2 ** (-1 / N * perplexity)


def sentence_prob(sentence, model):
    prob = 1
    for word in sentence.split():
        prob *= model[word]
    return prob


# here you construct the unigram language model
def unigram(tokens):
    stops = ['<', '>']
    model = collections.defaultdict(lambda: 0.001)
    for f in tokens:
        if f in stops:
            continue
        try:
            model[f] += 1
        except KeyError:
            model[f] = 1
            continue
    total = sum(model.values())
    for word in model:
        model[word] = model[word] / total

    model['<unk>'] = model['unk']
    del model['unk']
    return model


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

    datasets = OrderedDict()
    datasets['valid'] = PTB(
        data_dir=args.data_dir,
        split='valid',
        create_data=False,
        max_sequence_length=args.max_sequence_length,
        min_occ=args.min_occ
    )

    corpus = make_corpus(datasets['valid'])
    model = unigram(corpus)

    # Load samples
    samples_prime = load_prime_samples('samples/sents_sample_prime_n250.pkl')
    samples = np.load('samples/sents_sample_n250.pkl')
    samples_prime_dict = np.load('samples/sents_sample_prime_n250.pkl')

    # Compute and display perplexities
    print('Perplexity with constraints: {:.2f}'.format(perplexity(samples_prime, model)))
    print('Per tag:')
    for k in samples_prime_dict.keys():
        print('\t' + k + ': ', perplexity(samples_prime_dict[k], model))
    print('Perplexity without constraints: {:.2f}'.format(perplexity(samples, model)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--max_sequence_length', type=int, default=60)
    parser.add_argument('--min_occ', type=int, default=1)
    args = parser.parse_args()

    main(args)

