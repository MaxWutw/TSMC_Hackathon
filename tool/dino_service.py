from PIL import Image
import requests
from transformers import AutoProcessor, GroundingDinoForObjectDetection
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from flask import Flask, jsonify, request
from utils import convert_base64_to_image
import importlib.util
import os
import base64

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
device = torch.device("cpu")
print(f"Your device is: {device}")
processor_dino = AutoProcessor.from_pretrained("grounding-dino-base")
model_dino = GroundingDinoForObjectDetection.from_pretrained("grounding-dino-base").to(device)

app = Flask(__name__)
@app.route('/dino', methods=['POST'])
def predict():
    input_data = request.get_json()
    raw_image = input_data['img_base64']
    raw_image = convert_base64_to_image(raw_image)
    text = input_data['text']
    if '.' not in text:
        text+='.'
    inputs = processor_dino(images=raw_image, text=text, return_tensors="pt").to(device)
    outputs = model_dino(**inputs)
    target_sizes = torch.tensor([raw_image.size[::-1]])
    results = processor_dino.image_processor.post_process_object_detection(
        outputs, threshold=0.2, target_sizes=target_sizes
    )[0]
    bbox_list = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 1) for i in box.tolist()]
        bbox_list.append(box)
        print(f"Detected {label.item()} with confidence " f"{round(score.item(), 2)} at location {box}")
    return bbox_list

@app.route('/api', methods=['GET'])
def hello_world():
    return jsonify(message="Grounding Dino is running!")

if __name__ == '__main__':
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.py"))
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    app.run(host=config.IP, port=config.DINO_PORT, threaded=False)
