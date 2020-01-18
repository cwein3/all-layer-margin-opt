from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import torch
import os
import numpy as np
import torch.nn.functional as F

from PIL import Image

def load_data(batch_size, dataset, data_path="", corrupt_prob=0,data_size=50000, augment=True):
	normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
									 std=[x/255.0 for x in [63.0, 62.1, 66.7]])

	if augment:
		transform_train = transforms.Compose([
			transforms.ToTensor(),
			transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
								(4,4,4,4),mode='reflect').squeeze()),
			transforms.ToPILImage(),
			transforms.RandomCrop(32),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize
			])
	else:
		transform_train = transforms.Compose([
			transforms.ToTensor(),
			normalize
			])

	transform_test = transforms.Compose([
		transforms.ToTensor(),
		normalize
		])

	kwargs = {'num_workers': 1, 'pin_memory': True}

	if data_size == 50000:
		train_loader = torch.utils.data.DataLoader(
			RandomLabelDataset(dataset, corrupt_prob, train=True, transforms=transform_train),
			batch_size=batch_size, shuffle=True, **kwargs)
	else:
		train_data = RandomLabelDataset(dataset, corrupt_prob, train=True, transforms=transform_train)
		train_total = len(train_data)
		indices = list(range(train_total))
		np.random.shuffle(indices)
		train_indices = indices[:data_size]
		train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
		train_loader = torch.utils.data.DataLoader(
			dataset=train_data,
			batch_size=batch_size,
			sampler=train_sampler,
			**kwargs)

	val_loader = torch.utils.data.DataLoader(
		datasets.__dict__[dataset.upper()](data_path, train=False, transform=transform_test),
		batch_size=batch_size, shuffle=True, **kwargs)

	return train_loader, val_loader

class RandomLabelDataset(torch.utils.data.Dataset):
	def __init__(self, dataset, corrupt_prob, train=True, transforms=None):
		self.dataset = datasets.__dict__[dataset.upper()](
			'/tiger/u/colinwei/datasets', 
			train=train, 
			transform=transforms)

		self.corrupt_prob = corrupt_prob
		self.should_transform = np.random.random((len(self.dataset),))
		num_classes = 10 if dataset == 'cifar10' else 100
		self.rand_labels = np.random.choice(num_classes, len(self.dataset))

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		img, label = self.dataset[idx]
		if self.should_transform[idx] < self.corrupt_prob:
			label = int(self.rand_labels[idx])	
		return img, label