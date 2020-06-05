def get_model(path):
	from mnist_vae import VAE, MNISTConvEncoder, MNISTConvDecoder
	import torch

	modelvars = torch.load(path)

	model = VAE()
	model.load_state_dict(modelvars['state_dict'])	
	model.eval()
	model_dict = {'model':model}

	return model_dict

def decode(model, z):
	from torch import FloatTensor 
	import numpy
	# bijective (assuming sufficient disentanglemnt) map from real to model states
	# this needs to be derived by visual inspection of:
	# traversals.png for each model
	
	real2model = numpy.array([7, 2, 4, 9, 1, 8, 10, 5, 3, 6])

	z = numpy.array(z)
	
	z = numpy.array(z)
	z[5:] = z[5:][real2model-1]

	return model.decode(FloatTensor(z).unsqueeze(0)).cpu().numpy()