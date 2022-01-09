import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import copy

import requests
from PIL import Image, ImageDraw, ImageFont

# Utilize CUDA if available
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")

# Load Facebook DETR model
model = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=True)
model.eval()
#model = model.cuda()

# Mean and std from the ImageNet dataset
mean = np.array([0.485, 0.456, 0.496])
std = np.array([0.229, 0.224, 0.225])

# Standard PyTorch input image normalization
transform = transforms.Compose([
    transforms.Resize(800),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

CLASSES_SET = set(CLASSES)

def make_prediction(url):
    # Open image and transform into tensor
    img = Image.open(requests.get(url, stream=True).raw).resize((800,600))
    #img_tensor = transform(img).unsqueeze(0).cuda()
    img_tensor = transform(img).unsqueeze(0)
    
    # Makes prediction
    with torch.no_grad():
        output = model(img_tensor)
        
    pred_logits = output['pred_logits'][0][:, :len(CLASSES)]
    pred_boxes = output['pred_boxes'][0]

    # Perform softmax on predicted logits
    max_output = pred_logits.softmax(-1).max(-1)
    topk = max_output.values.topk(5)
    
    # Get only the top K logits and bounding boxes
    pred_logits = pred_logits[topk.indices]
    pred_boxes = pred_boxes[topk.indices]
    
    #img2 = img.copy()  # Make a copy of the original image for us to play with
    drw = ImageDraw.Draw(img)

    for logits, boxes in zip(pred_logits, pred_boxes):
        cls = logits.argmax() # Get the predicted class
        if cls >= len(CLASSES):
            continue
        label = CLASSES[cls] # Get the label
        # print(label)

        # Draw the bounding boxes on the image   
        boxes = boxes.cpu() * torch.Tensor([800, 600, 800, 600])
        x, y, w, h = boxes
        x0, x1 = x-w//2, x+w//2
        y0, y1 = y-h//2, y+h//2
        drw.rectangle([x0, y0, x1, y1], outline='red', width=3)

        #font_size = 16
        #font = ImageFont.truetype("arial.ttf", font_size)
        #drw.text((x0, y0), label, fill='white', font=font)

        drw.text((x0, y0), label, fill='white')


    return img




# Front End

st.title('Facebook DETR Object Detection App')

url = st.text_input("Input your image URL here")
if (url != ""):
    img = make_prediction(url)
    st.image(img)
# Input image address
#url = st.text_input("Input your image URL here", "https://images.fineartamerica.com/images/artworkimages/mediumlarge/2/ostrich-zebras-and-giraffe-garry-gay.jpg")
# url = input()

