from sklearn.cluster import KMeans
import numpy as np

from data_util import FDDB


all_boxes = []
fddb = FDDB()
for i in range(100):
	img, boxes = fddb.get_image_with_boxes(i)
	all_boxes.append( boxes[:,2:] )

all_boxes = np.concatenate(all_boxes, axis=0)

kmeans = KMeans(n_clusters=3)
kmeans.fit(all_boxes)
print(
	kmeans.cluster_centers_
)