from PIL import Image
import requests
from transformers import SamModel, SamProcessor
from transformers import pipeline
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from flask import Flask, jsonify, request
from utils import convert_base64_to_image
import importlib.util
import os
import base64
from torchvision.ops.boxes import batched_nms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Your device is: {device}")
generator = pipeline("mask-generation", model="sam-vit-large", device=device, points_per_batch=256)

app = Flask(__name__)
@app.route('/sam', methods=['POST'])
def sam_predict():
    input_data = request.get_json()
    raw_image = input_data['img_base64']
    raw_image = convert_base64_to_image(raw_image)
    results = generator(raw_image, points_per_batch=256)
    
    for mask in results["masks"]:
        print(mask)

    # return jsonify({"base64": mask_images_base64})

if __name__ == '__main__':
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.py"))
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    app.run(host=config.IP, port=config.API_PORT, threaded=False)
