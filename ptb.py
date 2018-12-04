from collections import defaultdict
from nltk.corpus import ptb, treebank
from string import punctuation as PUNCTUATION
from torch.utils.data import Dataset
import io
import json
import nltk
import numpy as np
import os
import torch
import time
from utils import OrderedCounter

class PTB(Dataset):

    def __init__(self, data_dir, split, create_data, **kwargs):

        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_sequence_length = kwargs.get('max_sequence_length', 50)
        self.min_occ = kwargs.get('min_occ', 3)

        self.raw_data_path = os.path.join(data_dir, 'ptb.'+split+'.txt')
        self.data_file = 'ptb.'+split+'.json'
        self.vocab_file = 'ptb.vocab.json'

        if create_data:
            print("Creating new %s ptb data."%split.upper())
            self._create_data()

        elif not os.path.exists(os.path.join(self.data_dir, self.data_file)):
            print("%s preprocessed file not found at %s. Creating new."%(split.upper(), os.path.join(self.data_dir, self.data_file)))
            self._create_data()

        else:
            self._load_data()


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        idx = str(idx)

        return {
            'input': np.asarray(self.data[idx]['input']),
            'input_str': self._get_str(self.data[idx]['input']),
            'input_tag': self._get_tag(self.data[idx]['input']),
            'target': np.asarray(self.data[idx]['target']),
            'target_str': self._get_str(self.data[idx]['target']),
            'target_tag': self._get_tag(self.data[idx]['target']),
            'length': self.data[idx]['length']
        }


    @property
    def vocab_size(self):
        return len(self.w2i)


    @property
    def pad_idx(self):
        return self.w2i['<pad>']


    @property
    def sos_idx(self):
        return self.w2i['<sos>']


    @property
    def eos_idx(self):
        return self.w2i['<eos>']


    @property
    def unk_idx(self):
        return self.w2i['<unk>']


    def get_w2i(self):
        return self.w2i


    def get_i2w(self):
        return self.i2w


    def _load_data(self, vocab=True):
        """loads data from a saves .json file, with or without a matching vocab"""
        with open(os.path.join(self.data_dir, self.data_file), 'r') as file:
            self.data = json.load(file)
        if vocab:
            with open(os.path.join(self.data_dir, self.vocab_file), 'r') as file:
                vocab = json.load(file)
            self.w2i, self.i2w = vocab['w2i'], vocab['i2w']


    def _get_str(self, idx_list):
        """for a given idx_list, uses i2w to return the original string"""
        output_strings = []
        for idx in idx_list:
            output_strings.append(self.i2w[str(idx)])

        return(output_strings)


    def _get_tag(self, idx_list):
        """for a given idx_list, uses """
        output_tags = []
        output_strings = self._get_str(idx_list)

        tags = nltk.pos_tag(output_strings)

        # skip the words
        for word, tag in tags:
            if word in ['<pad>', '<unk>', '<sos>', '<eos>']:
                output_tags.append('<unk>')
            else:
                output_tags.append(tag)

        return(output_tags)


    def _load_vocab(self):
        with open(os.path.join(self.data_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)

        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _is_number(self, string):
        try:
            float(string)
            return(True)
        except:
            return(False)


    def _is_key(self, dictionary, key):
        try:
            dictionary[key]
            return(True)
        except:
            return(False)


    def _preprocess(self, words):
        """removes punctuation, changes numbers to N, and non-vocab to <unk>"""
        output = []
        for word in words:
            if word in PUNCTUATION:
                pass
            elif self._is_number(word):
                output.append('N')
            elif not self._is_key(self.w2i, word.lower()):
                output.append('<unk>')
            else:
                output.append(word.lower())

        return(output)


    def _create_data(self):

        # hard coding of the number of samples for train and valid
        # n_train = 42069
        # n_valid = 7139
        # n_total = 49208

        if self.split == 'train':
            self._create_vocab()
        else:
            self._load_vocab()

        #tokenizer = TweetTokenizer(preserve_case=False)

        data = defaultdict(dict)

        # we build the dataset by looping through these inds of parsed_sents()
        if self.split == 'train':
            n_begin = 0
            n_end = 42069
        else:
            n_begin = 42069
            n_end = 49208

        # loop through treebank parsed sents
        for i in range(n_begin, n_end):

            t1 = time.time()
            words = ptb.parsed_sents()[i].leaves()
            words = self._preprocess(words)

            input = ['<sos>'] + words
            input = input[:self.max_sequence_length]

            target = words[:self.max_sequence_length-1]
            target = target + ['<eos>']

            assert len(input) == len(target), "%i, %i"%(len(input), len(target))
            length = len(input)

            input.extend(['<pad>'] * (self.max_sequence_length-length))
            target.extend(['<pad>'] * (self.max_sequence_length-length))

            input = [self.w2i.get(w, self.w2i['<unk>']) for w in input]
            target = [self.w2i.get(w, self.w2i['<unk>']) for w in target]

            id = len(data)
            data[id]['input'] = input
            data[id]['target'] = target
            data[id]['length'] = length

            t2 = time.time()
            print('sentence {}/{} done in {} sec'.format(i, n_end, t2-t1))

        with io.open(os.path.join(self.data_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)


    def _create_vocab(self):

        assert self.split == 'train', "Vocablurary can only be created for training file."

        #tokenizer = TweetTokenizer(preserve_case=False)

        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        with open(self.raw_data_path, 'r') as file:

            for i, line in enumerate(file):
                #words = tokenizer.tokenize(line)
                #words = nltk.word_tokenize(line)
                words = line.split()
                w2c.update(words)

            for w, c in w2c.items():
                if c > self.min_occ and w not in special_tokens:
                    i2w[len(w2i)] = w
                    w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)

        print("Vocablurary of %i keys created." %len(w2i))

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.data_dir, self.vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        self._load_vocab()

