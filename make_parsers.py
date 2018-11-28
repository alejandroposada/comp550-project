#!/usr/bin/env python

import chainer
from nltk import induce_pcfg, treetransforms
from nltk.corpus import ptb, treebank
from nltk.grammar import CFG, PCFG, Nonterminal, Production
from nltk.parse import ShiftReduceParser
from nltk.parse.viterbi import ViterbiParser
from nltk.probability import FreqDist
import nltk
import os
import pickle
import time

def graveyard():
    #LC_STRATEGY = [
    #    LeafInitRule(),
    #    FilteredBottomUpPredictCombineRule(),
    #    FilteredSingleEdgeFundamentalRule(),
    #]
    #parser = nltk.ChartParser(grammar, LC_STRATEGY, trace=2) # ??? time
    pass


def preprocess(item):
    return(Nonterminal(item.unicode_repr().split('-')[0].split('|')[0].split('+')[0].split('=')[0]))


def is_number(string):
    try:
        float(string)
        return(True)
    except:
        return(False)


def is_key(dictionary, key):
    try:
        dictionary[key]
        return(True)
    except:
        return(False)


def main():
    """
    makes a big dumb PTB CFG, and ShiftReduceParser, and a ViterbiParser, and
    serializes them all to disk for future use.

    The ViterbiParser runs in cubic time and give the most likely parse.
    The ShiftReduceParser runs in linear time and gives a single parse.

    https://stackoverflow.com/questions/7056996/how-do-i-get-a-set-of-grammar-rules-from-penn-treebank-using-python-nltk
    https://groups.google.com/forum/#!topic/nltk-users/_LXtbIekLvc
    https://www.nltk.org/_modules/nltk/grammar.html
    """
    vocabulary = chainer.datasets.get_ptb_words_vocabulary()
    freq_thresh = 0 ## ARBITRARY
    word_freqs = FreqDist(ptb.words())

    if not os.path.isfile('grammar.pkl'):

        productions = []
        add_dict = {}

        # use the entire treebank's parsed sentences to generate the CFG
        for i, tree in enumerate(ptb.parsed_sents()):

            # is it a good idea to combine this with my preprocessing?
            tree.collapse_unary(collapsePOS=False)
            tree.chomsky_normal_form(horzMarkov=2)

            # preprocess all productions by removing all tags
            these_productions = tree.productions()
            for production in these_productions:

                # remove all tags from the LHS (only keep primary tag)
                production._lhs = preprocess(production._lhs)

                rhs = []
                for item in production._rhs:

                    # remove all tags from the Nonterminals on the RHS
                    if type(item) == nltk.grammar.Nonterminal:
                        rhs.append(preprocess(item))

                    # replace numbers with N
                    elif is_number(item):
                        rhs.append('N')

                    # items not in dictionary replaced with <unk>
                    # dictionary requires lower
                    elif not is_key(vocabulary, item.lower()):
                        rhs.append('<unk>')

                    # replace infrequent words with <unk>
                    elif word_freqs[item] < freq_thresh:
                        rhs.append('<unk>')

                    # lowercase all entries in the grammar
                    else:
                        rhs.append(item.lower())

                production._rhs = tuple(rhs)

                if not is_key(add_dict, production.unicode_repr()):
                    add_dict[production.unicode_repr()] = True
                    productions.append(production)

        print('** {} productions found! **'.format(len(productions)))
        grammar = induce_pcfg(Nonterminal('S'), productions)

        with open('grammar.pkl', 'wb') as f:
            f.write(pickle.dumps(grammar))

    if not os.path.isfile('viterbi_parser.pkl'):
        filename = open('grammar.pkl', 'rb')
        grammar = pickle.load(filename)
        viterbi_parser = ViterbiParser(grammar, trace=0) # cubic time

        with open('viterbi_parser.pkl', 'wb') as f:
            f.write(pickle.dumps(viterbi_parser))

    if not os.path.isfile('shift_reduce_parser.pkl'):
        filename = open('grammar.pkl', 'rb')
        grammar = pickle.load(filename)
        shift_reduce_parser = ShiftReduceParser(grammar, trace=0)     # linear time

        with open('shift_reduce_parser.pkl', 'wb') as f:
            f.write(pickle.dumps(shift_reduce_parser))

    with open('data/ptb.train.txt', 'r') as f:
        data = f.readlines()

    #for sample in [1, 23, 20330, 20332, 443]:

    #    t1 = time.time()
    #    viterbi_parser.parse_one(data[sample].split())
    #    t2 = time.time()
    #    print('viterbi      = {:.2f} sec for {} words'.format(
    #        t2-t1, len(data[sample].split())))

    #    t1 = time.time()
    #    shift_reduce_parser.parse_one(data[sample].split())
    #    t2 = time.time()
    #    print('shift reduce = {:.2f} sec for {} words'.format(
    #        t2-t1, len(data[sample].split())))


if __name__ == '__main__':
    main()

