from mnist_vae import *
import torch
from viz.visualize import Visualizer
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np

bs = 64
gpu = True

model = VAE()
model = model.cuda() if gpu else model
model.train() 
print(model)
root 	  = os.path.join('mnist_data')

def get_mnist_dataloaders(batch_size=128, path_to_data='../data'):
    """MNIST dataloader with (32, 32) images."""
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    train_data = datasets.MNIST(path_to_data, train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST(path_to_data, train=False,
                               transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_loader, test_loader

data_loader, _ = get_mnist_dataloaders(bs, 'mnist_data')
optimizer = Adam(model.parameters(), lr=5e-4)		
iteration = 0

for e in range(100):
	
	epoch_loss = 0
	
	for data in data_loader:

		iteration += 1
		data = data[0].cuda() if gpu else data[0]

		optimizer.zero_grad()

		# forward 
		pred, z_pc, z_pd = model(data)
		
		# loss
		loss = model.loss(iteration, data, pred, z_pc, z_pd)

		# backward
		loss.backward()
		optimizer.step()
			
		epoch_loss += loss.item()
		
	
	print('Epoch {} Mean Loss - {}'.format(e+1,epoch_loss / len(data_loader.dataset)))
	print('Epoch: {} Average loss: {:.2f}'.format(e + 1,
														  64 * (32*32) *  (epoch_loss / len(data_loader.dataset))))

	save({'model': model, 'state_dict': model.state_dict()}, 'mnist_vae.pth')

model.eval()
viz = Visualizer(model)
samples = viz.samples()

traversals = viz.all_latent_traversals(size=10)

traversals = viz.latent_traversal_grid(cont_idx=2, cont_axis=1, disc_idx=0, disc_axis=0, size=(10, 10))
