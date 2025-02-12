from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from flask import Flask, jsonify, request
from utils import convert_base64_to_image
import importlib.util
import os
import base64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Your device is: {device}")
model = SamModel.from_pretrained("facebook/sam-vit-large")
processor_sam = SamProcessor.from_pretrained("facebook/sam-vit-large")

app = Flask(__name__)
@app.route('/sam', methods=['POST'])
def sam_predict():
    input_data = request.get_json()
    raw_image = input_data['img_base64']
    raw_image = convert_base64_to_image(raw_image)
    input_points = input_data['input_points']

    inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
    outputs = model(**inputs)
    masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
    scores = outputs.iou_scores

if __name__ == '__main__':
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.py"))
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    app.run(host=config.IP, port=config.DINO_PORT, threaded=False)
