import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

from data_loader import PublicDataset
from torch.utils.data import Dataset, DataLoader

from config import Config

class ConvNormRelu(nn.Module):
	def __init__(self, nin, nout, ksize, stride, padding):
		super().__init__()
		self.layers = nn.Sequential(
			nn.Conv2d(nin, nout, ksize, stride, padding),
			nn.BatchNorm2d(nout),
			nn.ReLU()
		)

	def forward(self, x):
		return self.layers(x)

class PositionalEmbedding(nn.Module):
	def __init__(self, T_max, dim):
		super().__init__()
		self.T_max = T_max
		self.dim = dim
		mat = (- 2 * np.arange(0, dim, 2) / dim * np.log(1e6))[:, None] +  np.log(np.arange(1, T_max + 1))[None, :]

		self.coss = torch.Tensor(np.cos(np.exp(mat)))
		self.sins = torch.Tensor(np.sin(np.exp(mat)))

		self.embeddings = torch.zeros(dim, T_max)
		self.embeddings[::2, :] = self.coss
		self.embeddings[1::2, :] = self.sins
		self.embeddings = self.embeddings.T # [T, dim],

	def forward(self, x):
		"""
		:param x: [batch, T, dim]
		:return:
		"""
		T = x.shape[1]
		return x + self.embeddings[:T, :]

class CTransformer(nn.Module):
	def __init__(self):
		super().__init__()

		""" CNN part """
		self.cnn = nn.Sequential(
			ConvNormRelu(3, 32, 3, 1, 1), # [batch, 3, 130, _]
			ConvNormRelu(32, 32, 3, 1, 1), # [batch, 32, 130, _]
			ConvNormRelu(32, 32, 2, 2, 0), # [batch, 32, 65, _] downsample by 2

			ConvNormRelu(32, 64, 3, 1, 1),  # [batch, 64, 65, _]
			ConvNormRelu(64, 64, 3, 1, 1),  # [batch, 64, 65, _]
			ConvNormRelu(64, 64, 2, 2, 0),  # [batch, 64, 32, _] downsample by 2

			ConvNormRelu(64, 128, 3, 1, 1),  # [batch, 128, 32, _]
			ConvNormRelu(128, 128, 3, 1, 1),  # [batch, 128, 32, _]
			ConvNormRelu(128, 128, 2, 2, 0),  # [batch, 128, 16, _] downsample by 2

			ConvNormRelu(128, 128, 3, 1, 1),# [batch, 128, 16, _]
			nn.Conv2d(128, 64, 1, 1, 0)# [batch, 64, 16, _]
		)
		self.downsample_factor = 8 # hard-coded in the CNN architecture
		self.cnn_out_channels = 64 # hard-coded in the CNN architecture
		self.cnn_out_height = 16

		""" Transformer part """
		self.pos_embed = PositionalEmbedding(T_max=1000, dim=1024)


		self.transformer = nn.Transformer(
			d_model=1024, nhead=4, num_encoder_layers=4, num_decoder_layers=4,
			dim_feedforward=512, dropout=0.1, activation='relu',batch_first=True
		)


	def cnn_forward(self, x, ws):
		"""
		:param x: torch.Tensor [batch, 3, 130, _]
		:param ws: The true width of the images before padding
		:return:
			x: torch.Tensor [batch, 128, 16, _]. The image features of padded images
			ws: The true width of the features (without padding)
		"""
		x = self.cnn(x)
		ws = ws // self.downsample_factor
		return x, ws

	def transformer_forward(self, x, ws):
		pass

	def forward(self, imgs, labels, ws, ls):
		"""
		:param imgs: torch.Tensor [batch, 3, 130, _]
		:param labels: torch.Tensor [batch, max_len, word_embed_dim]
		:param ws: The true width of the images before padding
		:param ls: The true length of the labels before padding
		:return:
		"""
		batch = imgs.shape[0]

		""" CNN part """
		x, ws = self.cnn_forward(imgs, ws)
		x = torch.reshape(x, [batch, self.cnn_out_height * self.cnn_out_channels, -1]) # [batch, 2048, T]


		""" Transformer part """
		x = torch.transpose(x, 1, 2) # [batch, T, 2048], T is the horizontal patches
		x = self.pos_embed(x) # plus positional embedding for transformer

		src_key_padding_mask = torch.zeros(batch, x.shape[1]).bool().to(Config.DEVICE) # ignore padding in src
		for i in range(batch):
			src_key_padding_mask[i, ws[i]:] = True

		trg_key_padding_mask = torch.zeros(batch, labels.shape[1]).bool().to(Config.DEVICE) # ignore padding in trg
		for i in range(batch):
			trg_key_padding_mask[i, ls[i]:] = True

		trg_mask = torch.tril(torch.ones(labels.shape[1], labels.shape[1]).bool()).bool().to(Config.DEVICE) # causal mask for trg
		preds = self.transformer(x, labels,
		                         tgt_mask=trg_mask,
		                         src_key_padding_mask=src_key_padding_mask,
		                         tgt_key_padding_mask=trg_key_padding_mask,
		                         memory_key_padding_mask=src_key_padding_mask)

		return preds # [batch, max_len, word_embed_dim]









def test_cnn():
	ds = PublicDataset()
	dl = DataLoader(ds, batch_size=3, shuffle=True, collate_fn=PublicDataset.collate_fn)
	ct = CTransformer()
	for imgs, labels, ws, ls in dl:

		# imgs = torch.permute(imgs, [0, 3, 1, 2])

		xs, ws = ct.cnn_forward(imgs, ws)
		print(xs.shape, ws)
		for i in range(3):
			fig, (ax1, ax2) = plt.subplots(1, 2)
			ax1.imshow(
				np.transpose(imgs[i].detach().numpy() * 255, [1, 2, 0]).astype(int)
			)
			ax2.imshow(xs[i, 0, :, :].detach().numpy())
			plt.show()
		break

def test_transformer():
	ds = PublicDataset()
	dl = DataLoader(ds, batch_size=3, shuffle=True, collate_fn=PublicDataset.collate_fn)
	ct = CTransformer()
	for imgs, labels, ws, ls in dl:
		labels = torch.randn(labels.shape[0], labels.shape[1], 1024) # fake embeddings
		ct.forward(imgs, labels, ws, ls)
		break


if __name__ == '__main__':
	# test_cnn()
	test_transformer()
