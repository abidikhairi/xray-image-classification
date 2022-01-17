import torch
import streamlit as st
from PIL import Image
from torchvision import transforms
from models import ResnetClassifier

MODEL_PATH = '/home/flursky/Work/xray-image-classification/data/resnetclassifier.pt'

LABEL_NAMES	= {
	0: 'NORMAL',
	1: 'PNEUMONIA'
}

def app():
	st.title('Image Prediction')
	image = st.file_uploader('Image')
	if image:
		img = Image.open(image).convert('RGB')
		
		result = run_predicition(img)
		label_name = result['Label']
		confidence = result['Confidence']

		st.image(image, caption='Uploaded Image')
		st.text(f'Label: {label_name}')
		st.text(f'Confidence: {confidence * 100:.2f} %')


def run_predicition(image):
	
	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=(0.225, 0.226, 0.267), std=(0.457, 0.456, 0.455))
	])

	tensor = transform(image).unsqueeze(0)

	model = ResnetClassifier()
	model.eval()
	model.load_state_dict(torch.load(MODEL_PATH))
	
	output = torch.exp(model(tensor)).squeeze()

	label = torch.argmax(output).item()
	label_name = LABEL_NAMES[label]
	confidence = output[label].item()

	result = {
		"Label": label_name,
		"Confidence": confidence
	}

	return result

