#!/usr/bin/env python

from nltk import induce_pcfg, treetransforms
from nltk.corpus import ptb, treebank
from nltk.grammar import CFG, PCFG, Nonterminal, Production
from nltk.parse import ShiftReduceParser
from nltk.parse.viterbi import ViterbiParser
from nltk.probability import FreqDist
import nltk
import os
import pickle

def graveyard():
    #LC_STRATEGY = [
    #    LeafInitRule(),
    #    FilteredBottomUpPredictCombineRule(),
    #    FilteredSingleEdgeFundamentalRule(),
    #]
    #parser = nltk.ChartParser(grammar, LC_STRATEGY, trace=2) # ??? time
    pass


def preprocess(item):
    return(Nonterminal(item.unicode_repr().split('-')[0].split('|')[0].split('+')[0]))


def is_number(string):
    try:
        float(string)
        return(True)
    except:
        return(False)


def is_added(add_dict, item):
    try:
        add_dict[item.unicode_repr()]
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
    freq_thresh = 30
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

                    # replace numbers with #
                    elif is_number(item):
                        rhs.append('#')

                    # replace infrequent words with <unk>
                    elif word_freqs[item] < freq_thresh:
                        rhs.append('<unk>')

                    else:
                        rhs.append(item)

                production._rhs = tuple(rhs)

                if not is_added(add_dict, production):
                    add_dict[production.unicode_repr()] = True
                    productions.append(production)

        grammar = induce_pcfg(Nonterminal('S'), productions)

        with open('grammar.pkl', 'wb') as f:
            f.write(pickle.dumps(grammar))

    if not os.path.isfile('viterbi_parser.pkl'):
        filename = open('grammar.pkl', 'rb')
        grammar = pickle.load(filename)
        parser = ViterbiParser(grammar, trace=0)                 # cubic time

        with open('viterbi_parser.pkl', 'wb') as f:
            f.write(pickle.dumps(parser))

    if not os.path.isfile('shift_reduce_parser.pkl'):
        filename = open('grammar.pkl', 'rb')
        grammar = pickle.load(filename)
        parser = ShiftReduceParser(grammar, trace=0)             # linear time

        with open('shift_reduce_parser.pkl', 'wb') as f:
            f.write(pickle.dumps(parser))


if __name__ == '__main__':
    main()

