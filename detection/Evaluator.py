import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt

from Detector import Detector, iou, nonmax_suppression, plot_prediction
from data_util import FDDB
from config import Params

from DetectionNetNaive import DetectionNetNaive
from DetectionNetMultiHead import DetectionNetMultiHead

class Evaluator():
	def __init__(self, model_cls, ckpt_path, device='cpu'):
		self.detector = Detector(model_cls, ckpt_path, device)
		self.dataset = FDDB(test=True)

	def predict_over_dataset(self):
		true_boxes_over_dataset = []
		predicted_boxes_over_dataset = []

		""" Get all predicted  """
		for i in range(len(self.dataset)):
		# for i in range(2):
			img, boxes_true = self.dataset.get_image_with_boxes(i)

			# add real bounding boxes to all boxes
			m1, _ = boxes_true.shape
			trues = np.concatenate([
				np.ones(shape=[m1, 1]) * i, boxes_true
			], axis=1)
			true_boxes_over_dataset.append(trues)

			img_tensor = np.transpose(img, [2,0,1])
			img_tensor = torch.from_numpy(img_tensor).float()[None,...] / 255

			pred_large, pred_mid, pred_small = self.detector.predict(img_tensor)
			preds = self.detector.filter_predictions(pred_large[0], pred_mid[0], pred_small[0])
			# plot_prediction(img, preds)

			# add the predicted bounding box if there's any
			if len(preds) != 0:
				m2,n = preds.shape
				preds = np.concatenate([
					np.ones(shape=[m2, 1]) * i, preds
				], axis=1)
				predicted_boxes_over_dataset.append(preds)

		# shape [m1, 5], each with [img_idx, x, y, w, h]
		true_boxes_over_dataset = np.concatenate(true_boxes_over_dataset, axis=0)
		# shape [m2, 6], each with [img_idx, confidence, x, y, w, h]
		predicted_boxes_over_dataset = np.concatenate(predicted_boxes_over_dataset, axis=0)
		return true_boxes_over_dataset, predicted_boxes_over_dataset

	def mean_average_precision(self, true_boxes_over_dataset, predicted_boxes_over_dataset):
		true_boxes_over_dataset = true_boxes_over_dataset.tolist()
		predict_boxes_over_dataset = predicted_boxes_over_dataset.tolist()
		num_true_boxes = len(true_boxes_over_dataset)

		# sort predictions with confidences
		predicted_boxes_over_dataset = sorted(predicted_boxes_over_dataset, key=lambda box:box[1], reverse=True)

		true_positive = 0
		false_positive = 0
		identified_labels = 0 # number of faces which are recognized by the model

		recalls = []
		precisions = []
		ious = []
		while len(predicted_boxes_over_dataset) != 0:
			box = predicted_boxes_over_dataset[0]
			predicted_boxes_over_dataset = predicted_boxes_over_dataset[1:]

			true_prediction = False
			corresponding_label_idx = -1
			for i, label in enumerate(true_boxes_over_dataset):
				if label[0] == box[0]:
					sample_iou = iou(box[2:], label[1:])
					if sample_iou  >= Params.IOU_THRESHOLD_MAP:
						# the predicted box is a true prediction
						true_prediction = True
						corresponding_label_idx = i
						ious.append(sample_iou)
						break

			if true_prediction: # if the predicted box corresponds to a label box
				true_positive += 1
				identified_labels += 1
				true_boxes_over_dataset.pop(corresponding_label_idx)
			else:
				false_positive += 1
				ious.append(0)

			recalls.append(identified_labels / num_true_boxes)
			precisions.append(true_positive / (true_positive + false_positive))

		ious = ious + [0 for _ in true_boxes_over_dataset]
		return recalls, precisions, ious

if __name__ == '__main__':

	e = Evaluator(DetectionNetNaive, Params.NAIVE_CKPT_PATH, Params.DEVICE)
	true_boxes, predicted_boxes = e.predict_over_dataset()
	recalls_naive, precisions_naive, ious_naive = e.mean_average_precision(true_boxes, predicted_boxes)
	map_naive = np.trapz(precisions_naive, recalls_naive)

	e = Evaluator(DetectionNetMultiHead, Params.MULTIHEAD_CKPT_PATH, Params.DEVICE)
	true_boxes, predicted_boxes = e.predict_over_dataset()
	recalls_multihead, precisions_multihead, ious_multihead = e.mean_average_precision(true_boxes, predicted_boxes)
	map_multihead = np.trapz(precisions_multihead, recalls_multihead)

	plt.plot(recalls_naive, precisions_naive, label='YoloV3 %.5f' % (map_naive * 100))
	plt.plot(recalls_multihead, precisions_multihead, label='Ours %.5f' % (map_multihead * 100))
	plt.legend()
	plt.savefig('eval.png')

	print('IOU YoloV3: %.6f' % (ious_naive))
	print('IOU Ours: %.6f' % (ious_multihead))



