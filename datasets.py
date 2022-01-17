import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class XRayImageDataset(Dataset):
	def __init__(self, filename, transform=None) -> None:
		super().__init__()
		self.frame = pd.read_csv(filename)
		self.transform = transform

	
	def __len__(self):
		return len(self.frame)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		label = self.frame.iloc[idx, 1]
		img_path = self.frame.iloc[idx, 0]

		img = Image.open(img_path).convert('RGB')

		if self.transform:
			img = self.transform(img)
		
		return img, label
