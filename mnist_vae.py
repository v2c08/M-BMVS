import os 
import dist 
from torch.nn import Module, Conv2d, ConvTranspose2d, Linear, Sigmoid
from torch.nn import functional as F
import torchvision.transforms.functional as TTF
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.optim import Adam
from torch import cat, zeros, sigmoid, save

class MNISTConvEncoder(Module):
	def __init__(self):
		super(MNISTConvEncoder, self).__init__()

		self.conv1 = Conv2d(1,	32, (4,4), stride=2, padding=1)
		self.conv2 = Conv2d(32, 64, (4,4), stride=2, padding=1)
		self.conv3 = Conv2d(64, 64, (4,4), stride=2, padding=1)

		self.fc1 	  = Linear(64 * 4 * 4, 256) 
		self.fc_zp    = Linear(256,10)
		self.fc_alpha = Linear(256, 10)

	def forward(self, x, z_q=None):
		
		latent_dist = {'con':[], 'dis':[]} 
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))

		h = F.relu(self.fc1(x.view(-1, 64 * 4 * 4)))

		latent_dist['cont'] = self.fc_zp(h)
		latent_dist['dis'] = F.softmax(self.fc_alpha(h), dim=1)

		return latent_dist['cont'], latent_dist['dis']

class MNISTConvDecoder(Module):
	def __init__(self):
		super(MNISTConvDecoder, self).__init__()
	
		self.fc1 = Linear(15, 256)
		self.fc2 = Linear(256,  64*4*4)
		
		self.deconv1 = ConvTranspose2d(64, 32, (4,4),stride=2,padding=1)
		self.deconv2 = ConvTranspose2d(32, 32, (4,4),stride=2,padding=1)
		self.deconv3 = ConvTranspose2d(32, 1,  (4,4),stride=2,padding=1)
		
		self.sig = Sigmoid()
		
	def forward(self, x):
		
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.deconv1(x.view(-1,64,4,4)))
		x = F.relu(self.deconv2(x))
		x = sigmoid(self.deconv3(x)) 
		return x

class VAE(Module):
	def __init__(self):
		super(VAE,self).__init__()
		
		self.latent_spec = {'cont': 5, 'disc': [10]}
		self.use_cuda = True
		# Initialise Distributions		
		self.q_dist	  = dist.Normal()
		self.cat_dist = dist.Gumbel(self.latent_spec['disc'], .67)

		self.latent_cont_dim = 0
		self.latent_disc_dim = 0
		self.num_disc_latents = 0		
		self.latent_cont_dim = self.latent_spec['cont']
		self.latent_disc_dim += sum([dim for dim in self.latent_spec['disc']])
		self.num_disc_latents = len(self.latent_spec['disc'])	
		self.latent_dim = self.latent_cont_dim + self.latent_disc_dim
			 
		self.f_enc = MNISTConvEncoder()
		self.g_dec = MNISTConvDecoder()

	def loss(self, curiter, image, pred, z_pc, z_pd, eval=False):

		flat_dim = 32 * 32	

		err_loss = F.binary_cross_entropy(pred.view(-1,flat_dim), image.view(-1,flat_dim))
		err_loss *= flat_dim

		
		kloss_args	= (z_pc,   # mu, sig
					   [0.0, 5.2, 25000, 30.0], # anealing params
					   curiter)	# data size
					   
		norm_kl_loss = self.q_dist.calc_kloss(*kloss_args)

		kloss_args	 = (z_pd,  # alpha
						[0.0, 5.2, 25000, 30.0],  # anneling params 
						[10], # nclasses per categorical dimension
						curiter)	# data size
					  
		cat_kl_loss = self.cat_dist.calc_kloss(*kloss_args)
		

		return (norm_kl_loss + cat_kl_loss + err_loss) / flat_dim
	
	def decode(self, z):
		return self.g_dec(z).data
	
	def forward(self, image):
			
		# Encoding - p(z2|x) or p(z1 |x,z2)
		z_pc, z_pd = self.f_enc(image, None)
		
		# Latent Sampling
		latent_sample = []

		norm_sample = self.q_dist.sample_normal(params=z_pc, train=self.training)
		latent_sample.append(norm_sample)

		cat_sample = self.cat_dist.sample_gumbel_softmax(z_pd, train=self.training)

		latent_sample.append(cat_sample)

		z = cat(latent_sample, dim=1)
		# Decoding - p(x|z)
		pred = self.g_dec(z)
		
		return pred, z_pc, [z_pd]
			
class MNISTransform(object):
	def __init__(self, b,t,dim):
		self.b = b
		self.t = t
		self.dim = dim
	def __call__(self, image):
		
		image = TTF.to_tensor(image) # (batch, c, h, w)
		new_im = zeros(image.shape[0],32,32)
		new_im[:,2:30,2:30] = image
		if self.t > 1:
			new_im = new_im.unsqueeze(1)	 # (batch, 1, c, h, w)
			new_im = new_im.expand(self.t,*self.dim) # (batch, t, c, h, w)
		return new_im		

