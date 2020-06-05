import math
import numpy as np
import torch
from torch import rand, log, zeros_like,  cat
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torch.autograd import Function

eps = 1e-12
np.random.seed(5)

class Gumbel(nn.Module):

	def __init__(self, n_classes, temperature):
		super(Gumbel, self).__init__()
		
		self.n_classes	 = n_classes
		self.temperature = temperature

	def sample_gumbel(self, shape, eps=1e-12):
		U = rand(shape).cuda()
		return -Variable(log(-log(U + eps) + eps))

	def sample_gumbel_softmax(self, alpha, train):

		if train:
			unif	  = torch.rand(alpha.size())
			if alpha.is_cuda:
				unif = unif.cuda()
			gumbel	  = -torch.log(-torch.log(unif + eps) + eps)
			log_alpha = torch.log(alpha + eps)
			logit	  = (log_alpha + gumbel) / self.temperature
			
			return F.softmax(logit, dim=1)
		else:
			_, max_alpha = torch.max(alpha, dim=-1)
			one_hot_samples = torch.zeros(alpha.size())
			one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)
			if alpha.is_cuda:
				return one_hot_samples.cuda()
			else:	
				return one_hot_samples


	def _recon_sample(self, alphas):
		
		_, max_alpha = torch.max(alphas, dim=-1)
		one_hot_sample = torch.zeros(alphas.size())
		#one_hot_sample.scatter_(1,max_alpha.view(-1,1).data.cpu(), 1).cuda()
		one_hot_sample.scatter_(1,max_alpha.view(-1,1).data.cpu(), 1)
		if alphas.is_cuda:
			return one_hot_sample.cuda()
		else:
			return one_hot_sample

	def calc_kloss(self, alphas, discap, disdim, cur_iter):

		kl_losses = [self._kloss(alpha) for alpha in alphas]

		kl_loss = torch.sum(cat(kl_losses)) 

		disc_min, disc_max, disc_num_iters, disc_gamma = discap
		# Increase discrete capacity without exceeding disc_max or theoretical
		# maximum (i.e. sum of log of dimension of each discrete variable)
		disc_cap_current = (disc_max - disc_min) * cur_iter / float(disc_num_iters) + disc_min
		disc_cap_current = min(disc_cap_current, disc_max)
		# Require float conversion here to not end up with numpy float
		disc_theoretical_max = sum([float(np.log(dim)) for dim in disdim])
		disc_cap_current = min(disc_cap_current, disc_theoretical_max)
		# Calculate discrete capacity loss
		disc_capacity_loss = disc_gamma * torch.abs(disc_cap_current - kl_loss)
		
		
		return disc_capacity_loss


	def _kloss(self, alpha):

		log_dim = torch.Tensor([np.log(int(alpha.size()[-1]))])
		
		if alpha.is_cuda:
			log_dim = log_dim.cuda()
		
		neg_entropy = torch.sum(alpha * torch.log(alpha + eps),dim=-1)
		mean_neg_entropy=torch.mean(neg_entropy, dim=0)
		kl_loss = log_dim + mean_neg_entropy
		return kl_loss

class STHeaviside(Function):
	@staticmethod
	def forward(ctx, x):
		y = torch.zeros(x.size()).type_as(x)
		y[x >= 0] = 1
		return y

	@staticmethod
	def backward(ctx, grad_output):
		return grad_output

class Normal(nn.Module):
	"""Samples from a Normal distribution using the reparameterization trick.
	"""

	def __init__(self, mu=0, sigma=1):
		super(Normal, self).__init__()
		self.normalization = Variable(torch.Tensor([np.log(2 * np.pi)]))

		self.mu = Variable(torch.Tensor([mu]))
		self.logsigma = Variable(torch.Tensor([math.log(sigma)]))

	def _check_inputs(self, size, mu_logsigma):
		if size is None and mu_logsigma is None:
			raise ValueError(
				'Either one of size or params should be provided.')
		elif size is not None and mu_logsigma is not None:

			mu = mu_logsigma.select(-1, 0).expand(size)
			logsigma = mu_logsigma.select(-1, 1).expand(size)
			return mu, logsigma
		elif size is not None:
			mu = self.mu.expand(size)
			logsigma = self.logsigma.expand(size)
			return mu, logsigma
		elif mu_logsigma is not None:
			mu = mu_logsigma.select(-1, 0)
			logsigma = mu_logsigma.select(-1, 1)
			return mu, logsigma
		else:
			raise ValueError(
				'Given invalid inputs: size={}, mu_logsigma={})'.format(
					size, mu_logsigma))
		

	def calc_kloss(self, params, cont_cap, cur_iter):
		
		mu, logsigma = torch.chunk(params, 2, dim=-1)

		kl_vals = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp())
		kl_means = torch.mean(kl_vals, dim=0)
		kl_loss = torch.sum(kl_means)
		cont_min, cont_max, cont_num_iters, cont_gamma = cont_cap
		# Increase continuous capacity without exceeding cont_max
		
		cont_cap_current = (cont_max - cont_min) * cur_iter / float(cont_num_iters) + cont_min
		cont_cap_current = min(cont_cap_current, cont_max)
		
		# Calculate continuous capacity loss
		return	cont_gamma * torch.abs(cont_cap_current - kl_loss)
	
	
	def sample_normal(self, params, train=True):
		
		mu, logsigma = torch.chunk(params, 2, dim=-1)

		if train:
	
			sigma = torch.exp(0.5 * logsigma)
			if mu.is_cuda:
				eps = torch.zeros(sigma.size()).normal_().cuda()
			else:
				eps = torch.zeros(sigma.size()).normal_()
			#logsigma = torch.clamp(logsigma, min=-103., max=87.)		
			sample = mu + sigma * eps 
			return sample
		else:
			return mu

	def log_density(self, sample, params=None):
		if params is not None:
			mu, logsigma = self._check_inputs(None, params)
		else:
			mu, logsigma = self._check_inputs(sample.size(), None)
			mu = mu.type_as(sample)
			logsigma = logsigma.type_as(sample)

		c = self.normalization.type_as(sample.data)
		
		retval = -.5 * (sample - mu) * (sample - mu) / (torch.exp(logsigma) * torch.exp(logsigma)) - 0.5 * torch.log(c*torch.exp(logsigma) * torch.exp(logsigma))
		
		return retval

	def NLL(self, params, sample_params=None):
		"""Analytically computes
			E_N(mu_2,sigma_2^2) [ - log N(mu_1, sigma_1^2) ]
		If mu_2, and sigma_2^2 are not provided, defaults to entropy.
		"""
		mu, logsigma = self._check_inputs(None, params)
		if sample_params is not None:
			sample_mu, sample_logsigma = self._check_inputs(None, sample_params)
		else:
			sample_mu, sample_logsigma = mu, logsigma

		c = self.normalization.type_as(sample_mu.data)
		nll = logsigma.mul(-2).exp() * (sample_mu - mu).pow(2) \
			+ torch.exp(sample_logsigma.mul(2) - logsigma.mul(2)) + 2 * logsigma + c
		return nll.mul(0.5)

	def kld(self, params):
		"""Computes KL(q||p) where q is the given distribution and p
		is the standard Normal distribution.
		"""
		mu, logsigma = self._check_inputs(None, params)
		# see Appendix B from VAE paper:
		# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
		# https://arxiv.org/abs/1312.6114
		# 0.5 * sum(1 + log(sigma^2) - mean^2 - sigma^2)
		kld = logsigma.mul(2).add(1) - mu.pow(2) - logsigma.exp().pow(2)
		kld.mul_(-0.5)
		return kld

	def get_params(self):
		return torch.cat([self.mu, self.logsigma])

	@property
	def nparams(self):
		return 2

	@property
	def ndim(self):
		return 1

	@property
	def is_reparameterizable(self):
		return True

	def __repr__(self):
		tmpstr = self.__class__.__name__ + ' ({:.3f}, {:.3f})'.format(
			self.mu.data[0], self.logsigma.exp().data[0])
		return tmpstr

