#!/usr/bin/env python

from sklearn.decomposition import PCA
from utils import load_pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn

def main(z, z_prime, name):

    z_interp = load_pickle(z)
    z_interp_prime = load_pickle(z_prime)
    n = z_interp.shape[0]

    clf = PCA(n_components=4)
    clf.fit(z_interp)

    all_z = []
    all_z.append(clf.transform(z_interp))
    all_z.append(clf.transform(z_interp_prime['NONE']))
    all_z.append(clf.transform(z_interp_prime['SBAR']))
    all_z.append(clf.transform(z_interp_prime['PP']))
    all_z.append(clf.transform(z_interp_prime['ADJP']))
    all_z.append(clf.transform(z_interp_prime['QP']))
    all_z.append(clf.transform(z_interp_prime['WHNP']))
    all_z.append(clf.transform(z_interp_prime['ADVP']))
    all_z = np.vstack(all_z)

    names = ['$z$']*n + ["$z$' NONE"]*n + ["$z$' SBAR"]*n + ["$z$' PP"]*n + ["$z$' ADJP"]*n + ["$z$' QP"]*n + ["$z$' WHNP"]*n + ["$z$' ADVP"]*n

    data = pd.DataFrame(all_z)
    data['names'] = names

    sns.scatterplot(
        x=0, y=1, data=data, hue='names', alpha=1, palette="cubehelix")

    plt.savefig('figs/latent_{}.png'.format(name))
    plt.savefig('figs/latent_{}.svg'.format(name))
    plt.close()

if __name__ == '__main__':
    main('samples/z_interp_n250.pkl', 'samples/z_interp_prime_n250.pkl', 'interp')
    main('samples/z_sample_n250.pkl', 'samples/z_sample_prime_n250.pkl', 'sample')

