import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from config import Params
from DetectionNetNaive import DetectionNetNaive
from DetectionNetMultiHead import DetectionNetMultiHead


def iou(box1, box2):
	xcenter1, ycenter1, w1, h1 = box1
	xcenter2, ycenter2, w2, h2 = box2

	x_upper_left = max(xcenter1 - w1/2, xcenter2 - w2/2)
	y_upper_left = max(ycenter1 - h1/2, ycenter2 - h2/2)
	x_lower_right = min(xcenter1 + w1/2, xcenter2 + w2/2)
	y_lower_right = min(ycenter1 + h1/2, ycenter2 + h2/2)

	w = max(x_lower_right - x_upper_left, 0)
	h = max(y_lower_right - y_upper_left, 0)

	inter = w * h
	union = w1 * h1 + w2 * h2 - inter

	iou = inter / union
	iou = max(iou,0)
	iou = min(iou,1)
	return iou

def nonmax_suppression(preds):
	confidences = preds[:,0]
	argsort = np.argsort(confidences)[::-1]
	preds = preds[argsort]
	for i in range(len(preds)):
		if preds[i,0] == 0:
			continue
		for j in range(i+1, len(preds)):
			if iou(preds[i,1:], preds[j,1:]) >= Params.IOU_THRESHOLD_NONMAX_SUPRESSION:
				preds[j,0] = 0

	confidences = preds[:, 0]
	return preds[confidences > 0.,:]

def plot_prediction(img, preds):
	fig, ax = plt.subplots(1)
	ax.imshow(img)
	for pred in preds:
		confidence, xcenter, ycenter, w, h = pred
		x1, y1 = xcenter - w / 2, ycenter - h / 2
		x2, y2 = xcenter + w / 2, ycenter + h / 2
		rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
		ax.add_patch(rect)
		confidence = min(confidence, 1.)
		ax.text(x1, y1, " %.2f" % (confidence), bbox=dict(facecolor='red', alpha=0.9))
	plt.show()

class Detector(object):
	# def __init__(self, device='cpu'):
	# 	self.ckpt_path = Params.NAIVE_CKPT_PATH
	# 	self.net = DetectionNetNaive(True, True, device=device)
	# 	self.device = device
	# 	self.img_size = Params.IMG_SIZE
	#
	# 	ckpt = torch.load(self.ckpt_path, map_location=torch.device(device))
	#
	# 	self.net.load_state_dict(ckpt['model_state_dict'], strict=False)

	def __init__(self, model_cls, ckpt_path, device='cpu'):
		self.ckpt_path = ckpt_path
		self.net = model_cls(True, True, device=device)
		self.device = device
		self.img_size = Params.IMG_SIZE
		ckpt = torch.load(self.ckpt_path, map_location=torch.device(device))
		self.net.load_state_dict(ckpt['model_state_dict'], strict=False)

	def filter_predictions(self, pred_large, pred_mid, pred_small):
		"""
		:param pred_large: np.array, of shape [5,14,14]
		:param pred_mid: np.array, of shape [5,28,28]
		:param pred_small: np.array, of shape [5,56,56]
		:return:
			np.array, of shape [n,5]
		"""
		preds = []
		for pred in [pred_large, pred_mid, pred_small]:
			confidence = pred[0, :, :]
			Is, Js = np.where(confidence >= Params.CONFIDENCE_THRESHOLD)
			for i in range(len(Is)):
				preds.append(
					pred[:, Is[i], Js[i]]
				)
		if len(preds) == 0:
			return []
		preds = np.array(preds)
		preds = nonmax_suppression(preds)
		return preds

	def predict(self, imgs):
		"""
		:param imgs: torch.Tensor, [n,3,448,448]
		:return:
		"""
		self.net.eval()
		with torch.no_grad():
			imgs = imgs.to(self.device)
			outputs_large, outputs_mid, outputs_small = self.net(imgs)
			predicts_large, predicts_mid, predicts_small = self.net.interpret_outputs(
				outputs_large, outputs_mid, outputs_small,
			)
		predicts_large = predicts_large.cpu().detach().numpy()  # [n, 5, 14, 14]
		predicts_mid = predicts_mid.cpu().detach().numpy()
		predicts_small = predicts_small.cpu().detach().numpy()

		return predicts_large, predicts_mid, predicts_small


	def predict_from_fnames(self, filenames):
		"""
		:param filenames:
		:type filenames:
		:return:
		:rtype:
		"""
		n = len(filenames)
		imgs = []
		for fname in filenames:
			img = cv2.imread(fname)
			img = img[:, :, [2, 1, 0]] / 255
			imgs.append(cv2.resize(img, (self.img_size, self.img_size)))
		imgs = np.array(imgs)
		imgs_tensor = imgs.transpose([0, 3, 1, 2]).astype(np.float32)
		imgs_tensor = torch.from_numpy(imgs_tensor).to(self.device)

		# outputs_large, outputs_mid, outputs_small = self.net(imgs_tensor)
		# predicts_large, predicts_mid, predicts_small = self.net.interpret_outputs(
		# 	outputs_large, outputs_mid, outputs_small,
		# )
		# predicts_large = predicts_large.cpu().detach().numpy() # [n, 5, 14, 14]
		# predicts_mid = predicts_mid.cpu().detach().numpy()
		# predicts_small = predicts_small.cpu().detach().numpy()

		predicts_large, predicts_mid, predicts_small = self.predict(imgs_tensor)
		# print(predicts_large.shape)
		for i in range(n):
			preds = self.filter_predictions(predicts_large[i], predicts_mid[i], predicts_small[i])
			plot_prediction(imgs[i], preds)

if __name__ == '__main__':
	# detector = Detector(model_cls=DetectionNetNaive, ckpt_path=Params.NAIVE_CKPT_PATH)
	# detector = Detector(model_cls=DetectionNetMultiHead, ckpt_path=Params.MULTIHEAD_CKPT_SYM_PATH)
	detector = Detector(model_cls=DetectionNetMultiHead, ckpt_path=Params.MULTIHEAD_CKPT_PATH)

	fnames = [
		'./img_tests/g10.jpg'
	]

	detector.predict_from_fnames(fnames)

