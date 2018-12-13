#!/usr/bin/env python

import numpy as np
from utils import load_pickle
from scipy.stats import chisquare

def main():

    tags = load_pickle('samples/tags_sample_n250.pkl')
    tags_prime = load_pickle('samples/tags_sample_prime_n250.pkl')
    t1 = chisquare(np.sum(tags_prime[0], axis=0), f_exp=np.sum(tags, axis=0))
    t2 = chisquare(np.sum(tags_prime[1], axis=0), f_exp=np.sum(tags_prime[0], axis=0))
    t3 = chisquare(np.sum(tags_prime[2], axis=0), f_exp=np.sum(tags_prime[0], axis=0))
    t4 = chisquare(np.sum(tags_prime[3], axis=0), f_exp=np.sum(tags_prime[0], axis=0))
    t5 = chisquare(np.sum(tags_prime[4], axis=0), f_exp=np.sum(tags_prime[0], axis=0))
    t6 = chisquare(np.sum(tags_prime[5], axis=0), f_exp=np.sum(tags_prime[0], axis=0))
    t7 = chisquare(np.sum(tags_prime[6], axis=0), f_exp=np.sum(tags_prime[0], axis=0))

    print('tags: z vs z prime                = {}'.format(t1))
    print('tags z_prime NONE vs z_prime SBAR = {}'.format(t2))
    print('tags z_prime NONE vs z_prime PP   = {}'.format(t3))
    print('tags z_prime NONE vs z_prime ADJP = {}'.format(t4))
    print('tags z_prime NONE vs z_prime QP   = {}'.format(t5))
    print('tags z_prime NONE vs z_prime WHNP = {}'.format(t6))
    print('tags z_prime NONE vs z_prime ADVP = {}'.format(t7))

    sample = load_pickle('samples/sents_sample_n250.pkl')
    sample_prime = load_pickle('samples/sents_sample_prime_n250.pkl')

    for a,b,c,d,e,f in zip(sample_prime['SBAR'], sample_prime['PP'],
        sample_prime['ADJP'], sample_prime['QP'], sample_prime['WHNP'], sample_prime['ADVP']):

        print('SBAR: {}'.format(a))
        print('PP:   {}'.format(b))
        print('ADJP: {}'.format(c))
        print('QP:   {}'.format(d))
        print('WHNP: {}'.format(e))
        print('ADVP: {}'.format(f))
        print('---')



if __name__ == '__main__':
    main()

