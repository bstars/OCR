import sys
sys.path.append('..')

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

from recognition.data_loader import PublicDataset
from recognition.tokenizer import Tokenizer
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
		batch, T, dim = x.shape
		return x + self.embeddings[:T, :dim]

class RecognitionModel(nn.Module):
	def __init__(self, tokenizer : Tokenizer, num_encoder_layer=6, num_decoder_layer=6):
		super().__init__()
		self.tokenizer = tokenizer
		self.cnn = nn.Sequential(
			ConvNormRelu(3, 32, 3, 1, 1),  # [batch, 3, 130, _]
			ConvNormRelu(32, 32, 3, 1, 1),  # [batch, 32, 130, _]
			ConvNormRelu(32, 32, 2, 2, 0),  # [batch, 32, 65, _] downsample by 2

			ConvNormRelu(32, 64, 3, 1, 1),  # [batch, 64, 65, _]
			ConvNormRelu(64, 64, 3, 1, 1),  # [batch, 64, 65, _]
			ConvNormRelu(64, 64, 2, 2, 0),  # [batch, 64, 32, _] downsample by 2

			ConvNormRelu(64, 128, 3, 1, 1),  # [batch, 128, 32, _]
			ConvNormRelu(128, 128, 3, 1, 1),  # [batch, 128, 32, _]
			ConvNormRelu(128, 128, 2, 2, 0),  # [batch, 128, 16, _] downsample by 2

			ConvNormRelu(128, 128, 3, 1, 1),  # [batch, 128, 16, _]
			nn.Conv2d(128, 64, 1, 1, 0)  # [batch, 64, 16, _]
		)
		self.downsample_factor = 8  # hard-coded in the CNN architecture
		self.cnn_out_channels = 64  # hard-coded in the CNN architecture
		self.cnn_out_height = 16 # hard-coded in the CNN architecture
		self.model_dim = self.cnn_out_height * self.cnn_out_channels

		""" Transformer part """
		self.pos_embed = PositionalEmbedding(T_max=1000, dim=self.model_dim)
		self.token_embed = nn.Embedding(1000, 1024)
		self.transformer = nn.Transformer(
			d_model=1024, nhead=4, num_encoder_layers=num_encoder_layer, num_decoder_layers=num_decoder_layer,
			dim_feedforward=512, dropout=0.1, activation='relu', batch_first=True
		)
		self.classification = nn.Sequential(
			nn.Linear(self.model_dim, 512), nn.ReLU(),
			nn.Linear(512, self.tokenizer.vocab_size)
		)

	def cnn_forward(self, imgs, img_widths):
		"""
		:param imgs: [batch, 3, 130, _]
		:param img_widths: [batch], original widths of images before padding
		:return:
		"""
		x = self.cnn(imgs)
		feature_widths = img_widths // self.downsample_factor
		return x, feature_widths



	def forward(self, imgs, labels, img_widths, label_lengths):
		"""
		:param imgs: torch.Tensor [batch, 3, 130, _]
		:param labels: torch.Tensor [batch, max_len, word_embed_dim]
		:param img_widths: The true width of the images before padding
		:param label_lengths: The true length of the labels before padding
		:return:
		"""
		batch = imgs.shape[0]


		""" CNN part """
		cnn_feature, cnn_feature_widths = self.cnn_forward(imgs, img_widths)
		cnn_feature = torch.reshape(cnn_feature, [batch, self.cnn_out_height * self.cnn_out_channels, -1]) # [batch, 2048, T]
		cnn_feature = torch.transpose(cnn_feature, 1, 2) # [batch, T, 1024]

		""" Transformer part"""
		cnn_feature = self.pos_embed(cnn_feature) # [batch, 1024, T]
		token_embedding = self.token_embed(labels) # [batch, max_len, 1024]
		token_embedding = self.pos_embed(token_embedding) # [batch, max_len, 1024]

		src_key_padding_mask = torch.zeros(batch, cnn_feature.shape[1]).bool().to(Config.DEVICE)  # ignore padding in src
		for i in range(batch):
			src_key_padding_mask[i, cnn_feature_widths[i]:] = True

		trg_key_padding_mask = torch.zeros(batch, labels.shape[1]).bool().to(Config.DEVICE)  # ignore padding in trg
		for i in range(batch):
			trg_key_padding_mask[i, label_lengths[i]:] = True

		trg_mask = torch.triu(torch.ones(labels.shape[1], labels.shape[1]), diagonal=1).bool().to(
			Config.DEVICE)  # causal mask for trg

		preds = self.transformer(cnn_feature, token_embedding,
		                         tgt_mask=trg_mask,
		                         src_key_padding_mask=src_key_padding_mask,
		                         tgt_key_padding_mask=trg_key_padding_mask,
		                         memory_key_padding_mask=src_key_padding_mask) # [batch, max_label_len, model_dim]
		logits = self.classification(preds) # [batch, max_label_len, vocab_size]
		# print(logits)
		return logits

	def generate(self, img, max_len=100):
		"""
		:param img: [3, 130, _]
		:return:
		:rtype:
		"""
		img = torch.Tensor(img).to(Config.DEVICE)[None,...]
		cnn_feature, _ = self.cnn_forward(img, np.array([img.shape[-1]]))
		cnn_feature = torch.reshape(cnn_feature, [1, self.cnn_out_height * self.cnn_out_channels, -1]) # [batch, 2048, T]
		cnn_feature = torch.transpose(cnn_feature, 1, 2)  # [1, T, 1024]

		cnn_feature = self.pos_embed(cnn_feature) # [1, T, 1024]
		cnn_feature = self.transformer.encoder(cnn_feature) # [1, T, 1024]

		# Autoregressive generation
		inputs = [self.tokenizer.char_to_idx['<BOW>']]
		inputs_embedding = self.token_embed(torch.LongTensor(inputs).to(Config.DEVICE)[None,...]) # [1, 1, 1024]


		while True:
			inputs_to_transformer = self.pos_embed(inputs_embedding)
			preds = self.transformer.decoder(inputs_to_transformer, cnn_feature)
			preds = self.classification(preds) # [1, 1, vocab_size]
			preds = torch.argmax(preds[0,-1], dim=-1).item()

			inputs.append(preds)
			if preds == self.tokenizer.char_to_idx['<EOW>'] or len(inputs) > max_len:
				return inputs
			new_input_embedding = self.token_embed(torch.LongTensor([preds]).to(Config.DEVICE)[None,...]) # [1, 1, 1024]
			inputs_embedding = torch.cat([inputs_embedding, new_input_embedding], dim=1) # [1, len, 1024]




def test_cnn():
	ds = PublicDataset()
	dl = DataLoader(ds, batch_size=3, shuffle=True, collate_fn=PublicDataset.collate_fn)
	ct = RecognitionModel()
	for imgs, labels, ws, ls in dl:


		xs, ws_ = ct.cnn_forward(imgs, ws)
		print(xs.shape, ws)
		for i in range(3):
			print(imgs[i].shape, ws[i])
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
	ct = RecognitionModel()
	for imgs, labels, ws, ls in dl:
		ct.forward(imgs, labels, ws, ls)
		break

def test_generate():
	recog = RecognitionModel()
	img = torch.rand(3, 130, 100)
	recog.generate(img)


if __name__ == '__main__':
	test_cnn()
	# test_transformer()
	# test_generate()


