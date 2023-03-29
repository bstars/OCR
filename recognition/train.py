import sys
sys.path.append('..')

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

from recognition.data_loader import PublicDataset
from recognition.tokenizer import Tokenizer
from recognition.recog_model import RecognitionModel
from config import Config, RecognitionConfig


def overfit():
	""" Overfit for sanity check """
	tokenizer = Tokenizer(RecognitionConfig.TOKENIZER_PATH)
	ds = PublicDataset(tokenizer)
	dl = DataLoader(ds, batch_size=6, shuffle=False, collate_fn=PublicDataset.collate_fn)
	model = RecognitionModel(tokenizer, num_encoder_layer=1, num_decoder_layer=1).to(Config.DEVICE)

	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	class_weights = np.ones(len(tokenizer.char_to_idx))
	class_weights[tokenizer.char_to_idx['<PAD>']] = 0.
	class_weights[tokenizer.char_to_idx['<EOW>']] = 0.8
	class_weights = torch.Tensor(class_weights).to(Config.DEVICE)
	criterion = nn.CrossEntropyLoss(reduction='mean', weight=class_weights)
	# criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=model.tokenizer.char_to_idx['<PAD>'])

	for imgs, labels, ws, ls in dl:
		img = imgs[0].detach().cpu().numpy()
		img = np.transpose(img * 255, [1, 2, 0]).astype(int)
		plt.imshow(img)
		plt.show()
		for i in range(1000):
			inputs = labels[:, :-1].to(Config.DEVICE)
			targets = labels[:, 1:].to(Config.DEVICE)
			logits = model(imgs.to(Config.DEVICE), inputs, ws, ls)
			loss = criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
			print(loss.item())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if i % 10 == 0:
				captions = model.generate(imgs[0].to(Config.DEVICE), max_len=30)
				print(model.tokenizer.decode(captions))



def train():
	tokenizer = Tokenizer(RecognitionConfig.TOKENIZER_PATH)
	ds_train = PublicDataset(tokenizer, split='train')
	dl_train = DataLoader(ds_train, batch_size=16, shuffle=True, collate_fn=PublicDataset.collate_fn)
	ds_val = PublicDataset(tokenizer, split='val')
	model = RecognitionModel(tokenizer, num_encoder_layer=6, num_decoder_layer=6).to(Config.DEVICE)

	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	class_weights = np.ones(len(tokenizer.char_to_idx))
	class_weights[tokenizer.char_to_idx['<PAD>']] = 0.
	class_weights[tokenizer.char_to_idx['<EOW>']] = 0.8
	class_weights = torch.Tensor(class_weights).to(Config.DEVICE)
	criterion = nn.CrossEntropyLoss(reduction='mean', weight=class_weights)
	# criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=model.tokenizer.char_to_idx['<PAD>'])

	for epoch in range(20):
		for i, (imgs, labels, img_widths, label_widths) in enumerate(dl_train):
			inputs = labels[:, :-1].to(Config.DEVICE)
			targets = labels[:, 1:].to(Config.DEVICE)
			logits = model(imgs.to(Config.DEVICE), inputs, img_widths, label_widths)
			loss = criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
			print("Epochs %d, iteration %d, loss %.4f" % (epoch, i, loss.item()))

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if i % 100 == 0:
				idx = np.random.randint(0, len(ds_val))
				img, label, img_width, label_width = ds_val[idx]
				captions = model.generate(img.to(Config.DEVICE), max_len=30)
				plt.imshow(img.detach().cpu().numpy().transpose([1, 2, 0]))
				plt.show()
				print(model.tokenizer.decode(captions))

		if epoch % 5 == 0:
			torch.save({
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()},
				'./%d.pth' % (epoch)
			)





if __name__ == '__main__':
	overfit()
	# train()
	# ds = PublicDataset()
	# img, _, _, _ = ds[0]
	# print(img)
	# print(
	# 	torch.triu(torch.ones(5, 5)).bool()
	# )