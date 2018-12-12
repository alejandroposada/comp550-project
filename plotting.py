import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from os import listdir


def to_np(df):
    return np.array(df['Value'])


def smooth(scalars, weight):
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


# Load records
records = {}
path = 'logs/to_plot/'
for file in listdir(path):
    if file[0] == '.' or file[-3:] == 'png':
        pass
    else:
        records[file[:-4]] = to_np(pd.read_csv(path + file))

records2 = {}
path2 = 'runs/to_plot/'
for file in listdir(path2):
    if file[0] == '.' or file[-3:] == 'png':
        pass
    else:
        records2[file[:-4]] = to_np(pd.read_csv(path2 + file))


def plot_(records_dict, name, path, smooth_weight, ylabel, xlabel='Epoch', ylim=None, figsize=(11, 6), epochs=100):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(np.linspace(0, epochs, len(records_dict[name])), smooth(records_dict[name], smooth_weight))
    ax.set_ylim(ylim)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    fig.savefig(path + name + '.png')


def plot_kl(valid_or_train):
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.linspace(0, 100, len(records[valid_or_train + '_kl']))
    l1 = ax.plot(x, smooth(records[valid_or_train + '_kl'], 0.7), label='KL divergence')
    ax2 = ax.twinx()
    l2 = ax2.plot(x, records['train_kl_weight'], label='KL weight', c='r')

    ax2.set_ylabel('KL weight')
    ax.set_ylabel('KL divergence')
    ax.set_xlabel('Epoch')

    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=8)
    ax.set_ylim([0, 15])
    fig.savefig(path + 'kl_' + valid_or_train + '.png')


# Make plots
plt.rc('text', usetex=True)
plt.rc('font', family='helvetica')
matplotlib.rcParams.update({'font.size': 17})

# VAE plots
plot_(records, 'valid_nll', path, 0.97, 'Log-likelihood')
#plot_('train_nll', 0.97, 'NLL')
plot_kl('valid')
plot_kl('train')

# Latent constraints plots
plot_(records2, 'g_loss', path2,  0, 'Loss')
plot_(records2, 'd_loss', path2, 0, 'Loss')
plot_(records2, 'distance_penalty', path2, 0.97, 'Distance penalty')


