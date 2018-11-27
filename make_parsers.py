from nltk import induce_pcfg, treetransforms
from nltk.corpus import ptb, treebank
from nltk.grammar import CFG, Nonterminal
from nltk.parse import ShiftReduceParser
from nltk.parse.chart import LeafInitRule, FilteredBottomUpPredictCombineRule, FilteredSingleEdgeFundamentalRule
from nltk.parse.viterbi import ViterbiParser
import nltk
import pickle

def graveyard():
    #LC_STRATEGY = [
    #    LeafInitRule(),
    #    FilteredBottomUpPredictCombineRule(),
    #    FilteredSingleEdgeFundamentalRule(),
    #]
    #parser = nltk.ChartParser(grammar, LC_STRATEGY, trace=2) # ??? time
    pass


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
    productions = []

    if not os.path.isfile('grammar.pkl'):

        # use the entire treebank's parsed sentences to generate the CFG
        for tree in ptb.parsed_sents():
            tree.collapse_unary(collapsePOS=False)
            tree.chomsky_normal_form(horzMarkov=1)
            productions += tree.productions()
        productions = list(set(productions))

        grammar = induce_pcfg(Nonterminal('S'), productions)

        with open('grammar.pkl', 'wb') as f:
            f.write(pickle.dumps(grammar))

    if not os.path.isfile('viterbi_parser.pkl'):
        filename = open('grammar.pkl', 'rb') as f
        grammar = pickle.load(filename)
        parser = ViterbiParser(grammar, trace=0)                 # cubic time

        with open('viterbi_parser.pkl', 'wb') as f:
            f.write(pickle.dumps(parser))

    if not os.path.isfile('shift_reduce_parser.pkl')
        filename = open('grammar.pkl', 'rb') as f
        grammar = pickle.load(filename)
        parser = ShiftReduceParser(grammar, trace=0)             # linear time

        with open('shift_reduce_parser.pkl', 'wb') as f:
            f.write(pickle.dumps(parser))


if __name__ == '__main__':
    main()

