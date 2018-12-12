from torch.nn.utils import spectral_norm
from utils import to_var
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


class LinearBatch(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(LinearBatch, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        out = self.linear(x)
        return self.batch_norm(out)


class LinearSpec(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(LinearSpec, self).__init__()
        self.linear = spectral_norm(nn.Linear(input_dim, output_dim, bias=bias))

    def forward(self, x):
        return self.linear(x)


class SentenceVAE(nn.Module):
    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
                sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, num_layers=1, bidirectional=False):
        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.latent_size = latent_size
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        elif rnn_type == 'lstm':
            raise NotImplementedError()
        else:
            raise ValueError()

        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)


    def encoder(self, x, sorted_lengths):
        """
        encodes x to produce a mean an log variance
        """
        bs = x.size(0)
        x_embed = self.embedding(x)
        x_packed = rnn_utils.pack_padded_sequence(
            x_embed, sorted_lengths.data.tolist(), batch_first=True)
        _, hidden = self.encoder_rnn(x_packed)

        # flatten hidden state
        if self.bidirectional or self.num_layers > 1:
            hidden = hidden.view(bs, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # returns mean, logv of hidden and context for the decoder
        return(x, x_embed, self.hidden2mean(hidden), self.hidden2logv(hidden))


    def reparameterize(self, batch_size, mean, logv):
        """
        uses mean + log variance to generate samples from a gaussian
        parameterized by those values
        """
        std = torch.exp(0.5 * logv) # std = logvar.mul(0.5).exp_() ??
        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z.mul(std).add_(mean)

        return(z)


    def decoder(self, x, x_embed, z, sorted_lengths, sorted_idx):
        """decodes a sentence from a set of samples """
        bs = x.size(0)
        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, bs, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # decoder input
        if self.word_dropout_rate > 0:

            # randomly replace decoder input with <unk>
            prob = torch.rand(x.size())

            if torch.cuda.is_available():
                prob=prob.cuda()

            prob[(x.data - self.sos_idx) * (x.data - self.pad_idx) == 0] = 1
            decoder_x = x.clone()
            decoder_x[prob < self.word_dropout_rate] = self.unk_idx
            x_embed = self.embedding(decoder_x)

        x_embed = self.embedding_dropout(x_embed)
        packed_x = rnn_utils.pack_padded_sequence(
            x_embed, sorted_lengths.data.tolist(), batch_first=True)

        # decoder forward pass
        x_tilde, _ = self.decoder_rnn(packed_x, hidden)

        # process outputs
        x_tilde = rnn_utils.pad_packed_sequence(x_tilde, batch_first=True)[0]
        x_tilde = x_tilde.contiguous()
        _, reversed_idx = torch.sort(sorted_idx)
        x_tilde = x_tilde[reversed_idx]
        b, s, _ = x_tilde.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(
            x_tilde.view(-1, x_tilde.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)

        return(logp)


    def forward(self, input_seq, length):

        batch_size = input_seq.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_seq = input_seq[sorted_idx]

        x, x_embed, mean, logvar = self.encoder(input_seq, sorted_lengths)
        z = self.reparameterize(batch_size, mean, logvar)
        logp = self.decoder(x, x_embed, z, sorted_lengths, sorted_idx)

        return(logp, mean, logvar, z)


    def inference(self, n=4, z=None):

        if z is None:
            batch_size = n
            z = to_var(torch.randn([batch_size, self.latent_size]))
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

        hidden = hidden.unsqueeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch which are still generating
        sequence_mask = torch.ones(batch_size, out=self.tensor()).byte()

        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long() # idx of still generating sequences with respect to current loop

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        t=0
        while(t<self.max_sequence_length and len(running_seqs)>0):

            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())

            input_sequence = input_sequence.unsqueeze(1)
            input_embedding = self.embedding(input_sequence)
            output, hidden = self.decoder_rnn(input_embedding, hidden)
            logits = self.outputs2vocab(output)
            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]
                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()
            t += 1

        return generations, z


    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.squeeze()

        # fix if sample is a single value, enforce it to be a 1D tensor
        if len(sample.shape) == 0:
            sample = sample.view(1)

        return sample


    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to


class Actor(nn.Module):
    def __init__(self, dim_z, dim_model, num_layers=4, num_labels=6, conditional_version=True):
        """
        Actor feed-forward network G(z) (figure 12a in the paper).
        :param dim_z: dimension of the VAE's latent vector.
        :param dim_model: number of units in the layers.
        :param num_layers: number of layers.
        :param num_labels: number of attribute labels to condition on.
        :param conditional_version: True if conditioning on attribute labels to compute G(z,y).
        """
        super(Actor, self).__init__()
        self.dim_z = dim_z
        self.dim_model = dim_model
        self.num_layers = num_layers
        self.conditional_version = conditional_version

        self.gate = nn.Sigmoid()
        if self.conditional_version:
            self.num_labels = num_labels
            self.cond_layer = LinearBatch(num_labels, dim_model)

        layers = []

        # In the cond. version, the labels are passed through a linear layer and are concatenated with z as the input
        input_dim = self.dim_z + self.dim_model if self.conditional_version else self.dim_z
        layers.append(LinearBatch(input_dim, self.dim_model))
        layers.append(nn.ReLU())

        for i in range(num_layers - 1):
            layers.append(LinearBatch(self.dim_model, self.dim_model))
            layers.append(nn.ReLU())
        layers.append(LinearBatch(self.dim_model, 2 * self.dim_z))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, label):
        z = x
        x = torch.cat((x, self.cond_layer(label)), dim=-1)
        o = self.layers(x)
        gate_input, dz = o.chunk(2, dim=-1)  # Half of the inputs are used as dz and the other half as gates
        gates = self.gate(gate_input)
        z_prime = (1 - gates) * z + gates * dz
        return z_prime


class Critic(nn.Module):
    def __init__(self, dim_z, dim_model, num_layers=4, num_outputs=1, num_labels=6, conditional_version=True):
        """
        Critic feed-forward network D(z) (figure 12b in the paper).
        :param dim_z: dimension of the VAE's latent vector.
        :param dim_model: number of units in the layers.
        :param num_layers: number of layers.
        :param num_labels: number of attribute labels to condition on.
        :param conditional_version: True if conditioning on attribute labels to compute D(z,y).
        """
        super(Critic, self).__init__()
        self.dim_z = dim_z
        self.dim_model = dim_model
        self.num_layers = num_layers
        self.conditional_version = conditional_version

        self.gate = nn.Sigmoid()
        if self.conditional_version:
            self.num_labels = num_labels
            self.cond_layer = LinearSpec(num_labels, dim_model)

        layers = []

        # In the cond. version, the labels are passed through a linear layer and are concatenated with z as the input
        input_dim = self.dim_z + self.dim_model if self.conditional_version else self.dim_z
        layers.append(LinearSpec(input_dim, self.dim_model))
        layers.append(nn.ReLU())

        for i in range(num_layers - 1):
            layers.append(LinearSpec(self.dim_model, self.dim_model))
            layers.append(nn.ReLU())
        layers.append(LinearSpec(self.dim_model, num_outputs))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, label=None):
        if self.conditional_version:
            x = torch.cat((x, self.cond_layer(label)), dim=-1)
        o = self.layers(x)
        return F.sigmoid(o)
