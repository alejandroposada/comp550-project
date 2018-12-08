import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class LinearBatch(nn.Module):
	def __init__(self, input_dim, output_dim, bias=True):
		super(Linear, self).__init__()
		self.linear = nn.Linear(input_dim, output_dim, bias=bias)
		self.batch_norm = nn.BatchNorm1d(output_dim)

	def forward(self, x):
		out = self.linear(x)
		return self.batch_norm(out)


class LinearSpec(nn.Module):
	def __init__(self, input_dim, output_dim, bias=True):
		super(Linear, self).__init__()
		self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.spec_norm = spectral_norm(output_dim)

	def forward(self, x):
		out = self.linear(x)
        return self.spec_norm(out)


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
