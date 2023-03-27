import sys
sys.path.append('..')

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

from data_loader import PublicDataset
from torch.utils.data import Dataset, DataLoader

from config import Config
from recognition.conv_trans import CTransformer
from recognition.tokenizer import Tokenizer


class RecognitionModel(nn.Module):
	def __init__(self, embedding_dim : int, tokenizer : Tokenizer):
		super().__init__()
		self.embedding_dim = embedding_dim
		self.tokenizer = tokenizer
		self.vocab_size = tokenizer.vocab_size

		self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
		self.ct = CTransformer()
		self.classification = nn.Linear(self.embedding_dim, self.vocab_size)


	def forward(self, imgs, labels, ws, ls):
		"""
		:param imgs: [batch, 3, 130, _]
		:param labels: [batch, T]
		:param ws: [batch, T]
		:param ls: [batch, T]
		:return:
		"""
		# [batch, T, embedding_dim]
		embeddings = self.token_embedding(labels)
		# [batch, T, embedding_dim]
		embeddings = self.ct(imgs, embeddings, ws, ls)
		# [batch, T, vocab_size]
		preds = self.classification(embeddings)
		return preds


	def generate(self, img):
		"""
		:param img: [1, 3, 130, _]
		:return:
		"""
		features = self.ct.cnn(img)
		# TODO: generate text at inference time

