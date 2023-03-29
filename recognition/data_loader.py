import sys
sys.path.append('..')

import os
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torchvision import transforms

from recognition.tokenizer import Tokenizer
from config import RecognitionConfig, Config


class PublicDataset(Dataset):
	"""
	Reshape the image to have a fixed height of Config.H
	Different image may have different widths, we will pad them to the same width as the widest image
	"""
	def __init__(self, tokenizer:Tokenizer, split='train'):
		self.path = '../datasets/ocr-datasets/public_dataset'
		self.tokenizer = tokenizer
		self.categories = ['Canon', 'Droid', 'E63', 'Palm', 'Reference']
		self.fnames = []
		self.labels = []
		if split == 'train':
			self.transforms = transforms.RandomApply(
				nn.ModuleList([
					transforms.GaussianBlur(3, sigma=(0.1, 0.15)),
					# transforms.RandomVerticalFlip(p=0.5),
				])
			)
		else:
			self.transforms = nn.Identity()

		for cate in self.categories[:1]:
			lines = open(
				os.path.join(self.path, cate + "_annotations.txt")
			).readlines()
			for l in lines:
				split = l.strip().split(" ")
				label = reduce(lambda x, y: x + y, split[1:], "")
				fname = os.path.join(self.path, "%s_cropped" % (cate), split[0].replace('_', '._'))
				img = plt.imread(fname)
				if img.shape[0] < img.shape[1]:
					self.fnames.append(
						os.path.join(self.path, "%s_cropped" %(cate), split[0].replace('_', '._'))
					)
					self.labels.append(label)
		idx = int(len(self.fnames) * 0.8)
		if split == 'train':
			self.fnames = self.fnames[:idx]
			self.labels = self.labels[:idx]
		else:
			self.fnames = self.fnames[idx:]
			self.labels = self.labels[idx:]
	def __getitem__(self, idx):
		img = plt.imread(self.fnames[idx])
		label = self.tokenizer.encode(self.labels[idx])
		h, w, _ = img.shape
		new_h, new_w = RecognitionConfig.H, int(w * RecognitionConfig.H / h)
		img = resize(img, [new_h, new_w])

		img = torch.Tensor(np.transpose(img, [2, 0, 1]))
		return self.transforms(img), torch.Tensor(label).long(), new_w, len(label)

	def __len__(self):
		return len(self.fnames)

	@staticmethod
	def collate_fn(batch):
		"""
		:param batch:
		:type batch:
		:return:
		"""
		ws = []
		ls = []
		imgs = []
		labels = []
		for img, label, w, l in batch:
			# print(img.shape, label.shape, w)
			ws.append(w)
			ls.append(l)
			imgs.append(img)
			labels.append(label)

		maxw = max(ws)
		imgs = [
			F.pad(img, [0, maxw - img.shape[-1]]) for img in imgs
		]
		labels = pad_sequence(labels, batch_first=True, padding_value=1)

		return torch.stack(imgs), labels, np.array(ws), np.array(ls)

def test_dataset():
	ds = PublicDataset()
	for i in range(10):
		img, label, w, l = ds[i]
		img = img.detach().numpy()
		img = np.transpose(img, [1, 2, 0])
		caption = ds.tokenizer.decode(label)

		plt.imshow(img)
		plt.title(caption)
		plt.show()



if __name__ == '__main__':
	# ds = PublicDataset()
	# dl = DataLoader(ds, batch_size=3, shuffle=True, collate_fn=PublicDataset.collate_fn)
	# for imgs, labels, ws, ls in dl:
	# 	print(ls)
	# 	for label in labels:
	# 		print(ds.tokenizer.decode(np.array(label).astype(int)))
	# 	break
	test_dataset()



	



