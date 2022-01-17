import argparse
import os
import torch
import torchmetrics.functional as metrics
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from models import ResnetClassifier
from datasets import XRayImageDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, criterion, optimizer, loader):
	model.train()

	total_loss = []
	total_acc = []

	for images, labels in loader:
		optimizer.zero_grad()

		images = images.to(device)
		labels = labels.to(device)

		logits = model(images)

		loss = criterion(logits, labels)
		accuracy = metrics.accuracy(logits, labels)

		loss.backward()
		optimizer.step()

		total_loss.append(loss.item())
		total_acc.append(accuracy.item())

	return torch.tensor(total_loss).mean(), torch.tensor(total_acc).mean()


def evaluate(model, criterion, loader):
	model.eval()

	with torch.no_grad():
		total_loss = []
		total_acc = []

		for images, labels in loader:		
			images = images.to(device)
			labels = labels.to(device)

			logits = model(images)

			loss = criterion(logits, labels)
			accuracy = metrics.accuracy(logits, labels)

			total_loss.append(loss.item())
			total_acc.append(accuracy.item())

	return torch.tensor(total_loss).mean(), torch.tensor(total_acc).mean()

def validate(model, criterion, loader):
	model.eval()

	with torch.no_grad():
		total_loss = []
		total_acc = []

		for images, labels in loader:		
			images = images.to(device)
			labels = labels.to(device)

			logits = model(images)

			loss = criterion(logits, labels)
			accuracy = metrics.accuracy(logits, labels)

			total_loss.append(loss.item())
			total_acc.append(accuracy.item())

	return torch.tensor(total_loss).mean(), torch.tensor(total_acc).mean()

def main(args):
	batch_size = args.batch_size
	num_epochs = args.num_epochs
	learning_rate = args.learning_rate
	model_path = args.model_path
	
	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=(0.225, 0.226, 0.267), std=(0.457, 0.456, 0.455))
	])
	
	trainset = XRayImageDataset('./data/train.csv', transform=transform)
	testset = XRayImageDataset('./data/test.csv', transform=transform)
	validset = XRayImageDataset('./data/val.csv', transform=transform)

	trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
	testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
	validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=1)

	model = ResnetClassifier()
	criterion = nn.NLLLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	model.to(device)

	for epoch in range(num_epochs):
		print(f'Starting Epoch {epoch + 1}/{num_epochs}')
		train_loss, train_accuracy = train(model, criterion, optimizer, trainloader)
		eval_loss, eval_accuracy = evaluate(model, criterion, testloader)
		valid_loss, valid_accuracy = validate(model, criterion, validloader)

		print(f'Training Step ->\tLoss: {train_loss:.2f}\tAccuracy: {train_accuracy * 100:.4f} %')
		print(f'Evaluation Step ->\tLoss: {eval_loss:.2f}\tAccuracy: {eval_accuracy * 100:.4f} %')
		print(f'Validation Step ->\tLoss: {valid_loss:.2f}\tAccuracy: {valid_accuracy * 100:.4f} %')

	torch.save(model.state_dict(), os.path.join(model_path, model._get_name().lower() + '.pt'))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()


	parser.add_argument('--batch-size', type=int, default=64, help='batch size (Default: 64)')
	parser.add_argument('--learning-rate', type=float, default=1e-4, help='learning rate (Default: 0.0001')
	parser.add_argument('--num-epochs', type=int, default=10, help='number of training epochs (Default: 10)')
	parser.add_argument('--model-path', type=str, required=True, help='model dir')

	args = parser.parse_args()

	main(args)