from collections import defaultdict, Counter, OrderedDict
from nltk import induce_pcfg
from nltk import treetransforms
from nltk.corpus import ptb
from nltk.corpus import treebank
from nltk.grammar import CFG, Nonterminal
from nltk.parse import ShiftReduceParser
from nltk.parse.viterbi import ViterbiParser
from torch.autograd import Variable
import nltk
import numpy as np
import time
import torch


class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def idx2word(idx, i2w, pad_idx):
    sent_str = [str()]*len(idx)

    for i, sent in enumerate(idx):
        for word_id in sent:
            if word_id == pad_idx:
                break

            # call word_id.item() to do proper conversion into str
            sent_str[i] += i2w[str(word_id.item())] + " "

        sent_str[i] = sent_str[i].strip()

    return(sent_str)


def interpolate(start, end, steps):

    interpolation = np.zeros((start.shape[0], steps + 2))

    for dim, (s,e) in enumerate(zip(start,end)):
        interpolation[dim] = np.linspace(s,e,steps+2)

    return interpolation.T


def expierment_name(args, ts):

    exp_name = str()
    exp_name += "BS=%i_"%args.batch_size
    exp_name += "LR={}_".format(args.learning_rate)
    exp_name += "EB=%i_"%args.embedding_size
    exp_name += "%s_"%args.rnn_type.upper()
    exp_name += "HS=%i_"%args.hidden_size
    exp_name += "L=%i_"%args.num_layers
    exp_name += "BI=%i_"%args.bidirectional
    exp_name += "LS=%i_"%args.latent_size
    exp_name += "WD={}_".format(args.word_dropout)
    exp_name += "ANN=%s_"%args.anneal_function.upper()
    exp_name += "K={}_".format(args.k)
    exp_name += "X0=%i_"%args.x0
    exp_name += "TS=%s"%ts

    return exp_name


def get_parse(idx):
    tree = ptb.parsed_sents()[idx]
    tree.pprint()


def find_parse_tag(tag):
    pass


def generate_parse_tree(sentence):
    pass


def evaluate_parse_quality(parse):
    pass


def check_grammar(grammar, sentence):
    grammar.check_coverage(sentence.split())


