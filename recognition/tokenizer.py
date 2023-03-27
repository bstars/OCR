import numpy as np
import scipy
from functools import reduce

class Tokenizer():

	@staticmethod
	def build():

		annotation_files = [
			('../datasets/ocr-datasets/COCO-Text-words-trainval/train_words_gt.txt', ','),
			('../datasets/ocr-datasets/COCO-Text-words-trainval/val_words_gt.txt', ','),
			('../datasets/ocr-datasets/public_dataset/Canon_annotations.txt', ' '),
			('../datasets/ocr-datasets/public_dataset/Droid_annotations.txt', ' '),
			('../datasets/ocr-datasets/public_dataset/E63_annotations.txt', ' '),
			('../datasets/ocr-datasets/public_dataset/Palm_annotations.txt', ' '),
			('../datasets/ocr-datasets/public_dataset/Reference_annotations.txt', ' ')
		]
		unique_chars = set()
		for fname, split_char in annotation_files:
			file = open(fname)
			lines = file.readlines()

			for l in lines:
				split = l.strip().split(split_char)
				label = reduce(lambda x, y: x + y, split[1:], "")
				unique_chars = unique_chars.union(set(label))

		print(unique_chars)
		file = open('../datasets/tokenizer.txt', 'w')
		# by default the space is 4
		file.write("%s %d\n" % (" ", 0))
		file.write("%s %d\n" % ("<PAD>", 1))
		file.write("%s %d\n" % ("<BOW>", 2))
		file.write("%s %d\n" % ("<EOW>", 3))
		file.write("%s %d\n" % ("<UNK>", 4))
		i = 5
		for c in sorted(list(unique_chars)):
			if c != " ":
				file.write("%c %d\n" % (c, i))
				i += 1

	def __init__(self, fname):
		self.idx_to_char = [' ']
		self.char_to_idx = dict()
		self.char_to_idx[' '] = 0
		file = open(fname)
		for line in file.readlines()[1:]:
			c, i = line.strip().split(' ')
			self.idx_to_char.append(c)
			self.char_to_idx[c] = int(i)

		self.vocab_size = len(self.idx_to_char)

	def encode(self, word):
		ret = []
		for c in word:
			ret.append(
				self.char_to_idx.get(c, 'UNK')
			)
		ret = [self.char_to_idx["<BOW>"]] + ret + [self.char_to_idx["<EOW>"]]
		return np.array(ret)

	def decode(self, idx):
		ret = ""
		for i in idx:
			ret += self.idx_to_char[i]
		return ret


def preprocess_coco(split='train'):
	file = open('../datasets/ocr-datasets/COCO-Text-words-trainval/%s_words_gt.txt' % (split))
	lines = file.readlines()
	for l in lines:
		split = l.strip().split(',')
		fname = split[0]
		label = reduce(lambda x, y : x + y, split[1:], "")
		# label.replace('\\\'', '\'')
		print(fname, label)


if __name__ == '__main__':
	Tokenizer.build()
	tokenizer = Tokenizer('../datasets/tokenizer.txt')

	encode = tokenizer.encode(" hello")
	print(encode)
	print(tokenizer.decode(encode))

