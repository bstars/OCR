import torch.optim
import numpy as np
from torch.utils.data import DataLoader
import os

from DetectionNetAbstract import DetectionNet
from DetectionNetMultiHead import DetectionNetMultiHead
from data_util import FDDB, plot_img_with_label
from config import Params

def train_naive(model:DetectionNet,
				training_loader:DataLoader,
				testing_set:FDDB,
				learning_rate,
				epoches,
				device='cpu',
				checkpoint=None,
                restore_optimizer=False):
	model.to(device)
	optimizer = torch.optim.Adam(
		filter( lambda p : p.requires_grad, model.parameters()),
		lr = learning_rate, weight_decay=1e-4
	)

	if checkpoint is not None:
		model.load_state_dict(checkpoint['model_state_dict'])
		if restore_optimizer:
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])



	for e in range(epoches):
		for i, (img, label_large, label_mid, label_small) in enumerate(training_loader):
			model.train()
			img = img.to(device)
			label_large = label_large.to(device)
			label_mid = label_mid.to(device)
			label_small = label_small.to(device)

			optimizer.zero_grad()
			output_large, output_mid, output_small = model(img)
			loss = model.compute_loss(output_large, output_mid, output_small, label_large, label_mid, label_small)
			loss.backward()
			optimizer.step()

			print("%d epoch, %d/%d, loss=%.5f" % (e, i, len(training_loader), loss.item()))

			if i % 100 == 0:

				pred_large, pred_mid, pred_small = model.interpret_outputs(output_large, output_mid, output_small)
				idx = np.random.randint(0, pred_large.shape[0], 1)[0]
				plot_img_with_label(
					img[idx].cpu().detach().numpy().transpose([1, 2, 0]),
					pred_large[idx].cpu().detach().numpy(),
					pred_mid[idx].cpu().detach().numpy(),
					pred_small[idx].cpu().detach().numpy()
				)

				model.eval()
				idx = np.random.randint(0, len(testing_set), 1)[0]
				img_tensor, _, _, _ = testing_set[idx]
				img_tensor = img_tensor.to(device)
				output_large, output_mid, output_small = model(img_tensor[None,...])
				pred_large, pred_mid, pred_small = model.interpret_outputs(output_large, output_mid, output_small)


				plot_img_with_label(
					img_tensor.cpu().detach().numpy().transpose([1, 2, 0]),
					pred_large[0].cpu().detach().numpy(),
					pred_mid[0].cpu().detach().numpy(),
					pred_small[0].cpu().detach().numpy()
				)
		if e % 10 == 0:
			torch.save({
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()},
				os.path.join(Params.CHECKPOINTS_SAVING_PATH, "%d.pth" % (e)))

def overfit(model:DetectionNet, training_loader, device):
	optimizer = torch.optim.Adam(
		filter(lambda p : p.requires_grad, model.parameters()),
		lr=1e-4
	)
	for img, label_large, label_mid, label_small in training_loader:

		for _ in range(50):
			img = img.to(device)
			label_large = label_large.to(device)
			label_mid = label_mid.to(device)
			label_small = label_small.to(device)

			optimizer.zero_grad()
			output_large, output_mid, output_small = model(img)
			loss = model.compute_loss(output_large, output_mid, output_small, label_large, label_mid, label_small)
			loss.backward()
			optimizer.step()
			print(loss.item())

			pred_large, pred_mid, pred_small = model.interpret_outputs(output_large, output_mid, output_small)
			idx = np.random.randint(0, pred_mid.shape[0],1)[0]

			plot_img_with_label(
				img[idx].cpu().detach().numpy().transpose([1,2,0]),
				pred_large[idx].cpu().detach().numpy(),
				pred_mid[idx].cpu().detach().numpy(),
				pred_small[idx].cpu().detach().numpy()
			)

		break
	torch.save({'model_state_dict' : model.state_dict()}, 'overfit.pth')

def check_overfit(model:DetectionNet, training_loader, device):
	model.eval()
	for img, label_large, label_mid, label_small in training_loader:
		img = img.to(device)
		label_large = label_large.to(device)
		label_mid = label_mid.to(device)
		label_small = label_small.to(device)

		for _ in range(10000):


			output_large, output_mid, output_small = model(img)
			loss = model.compute_loss(output_large, output_mid, output_small, label_large, label_mid, label_small)

			print(loss.item())

			pred_large, pred_mid, pred_small = model.interpret_outputs(output_large, output_mid, output_small)
			idx = np.random.randint(0, pred_mid.shape[0],1)[0]

			plot_img_with_label(
				img[idx].cpu().detach().numpy().transpose([1,2,0]),
				pred_large[idx].cpu().detach().numpy(),
				pred_mid[idx].cpu().detach().numpy(),
				pred_small[idx].cpu().detach().numpy()
			)

		break
	torch.save({'model_state_dict' : model.state_dict()}, 'overfit.pth')

if __name__ == '__main__':
	training_loader = DataLoader(FDDB(False), Params.BATCH_SIZE, shuffle=False, num_workers=4)

	testing_set = FDDB(test=True)
	model = DetectionNetMultiHead(True, True, Params.DEVICE)

	checkpoint = torch.load(Params.MULTIHEAD_CKPT_PATH)

	train_naive(model, training_loader, testing_set, Params.SEMI_LEARNING_RATE, 120, Params.DEVICE, checkpoint, False)

	model.load_state_dict(checkpoint['model_state_dict'])
	pass


	# train_naive(
	# 	model, training_loader, testing_set, Params.LEARNING_RATE, epoches=200, device=Params.DEVICE
	# )

	# overfit(model, training_loader, Params.DEVICE)






