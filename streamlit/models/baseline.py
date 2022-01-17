from torchvision import models
from torch import nn


class ResnetClassifier(nn.Module):
	def __init__(self):
		super(ResnetClassifier, self).__init__()
		
		self.resnet = models.resnet18(pretrained=True)
		self.classifier = nn.Sequential(
			nn.Linear(in_features=1000, out_features=512, bias=True),
			nn.Dropout(p=0.6),
			nn.Linear(in_features=512, out_features=2),
			nn.LogSoftmax(dim=1)	
		)

	def forward(self, images):
		z = self.resnet(images)

		return self.classifier(z)
