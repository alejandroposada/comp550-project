#!/usr/bin/env python

from nltk.tokenize import word_tokenize
import nltk


with open('ptb.train.txt', 'r') as f:
    data = f.readlines()

all_text = []
for i in range(len(data)):
    all_text.extend(word_tokenize(data[i]))
tags = nltk.pos_tag(all_text)

tag_fd = nltk.FreqDist(tag for (word, tag) in tags)

print(tag_fd)

import IPython; IPython.embed()
