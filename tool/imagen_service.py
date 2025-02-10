# https://cloud.google.com/vertex-ai/generative-ai/docs/samples/generativeaionvertexai-imagen-generate-image

import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from vertexai.preview.vision_models import Image as vertexImage
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
import random
from flask import Flask, jsonify, request
from utils import convert_base64_to_image, get_base64
import os
import importlib.util
import base64

app = Flask(__name__)
@app.route('/imagen', methods=['POST'])
def generate():
    output_file = "output_img/input-image1.png"

    input_data = request.get_json()
    text = input_data['text']
    # text = text + ".Please generate the image base on the below image(which had been turn into base64):" + raw_image
    if '.' not in text:
        text+='.'
    model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")

    images = model.generate_images(
        prompt=text,
        # Optional parameters
        number_of_images=1,
        language="en",
        # You can't use a seed value and watermark at the same time.
        # add_watermark=False,
        # seed=100,
        aspect_ratio="1:1",
        safety_filter_level="block_some",
        # person_generation="allow_adult",
    )

    images[0].save(location=output_file, include_generation_parameters=False)

    # Optional. View the generated image in a notebook.
    # images[0].show()

    print(f"Created output image using {len(images[0]._image_bytes)} bytes")
    with open(output_file, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

@app.route('/api', methods=['GET'])
def check():
    return jsonify(message="Imagen is running!")

if __name__ == '__main__':
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.py"))
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    PROJECT_ID = "tsmccareerhack2025-aaid-grp2"
    credential_path = "../va/tsmccareerhack2025.json"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    # output_file = "output_img/input-image2.png"
    # prompt = "Please draw me 3 dogs"

    vertexai.init(project=PROJECT_ID, location="us-central1")
    app.run(host=config.IP, port=config.IMAGEN_PORT, threaded=False)

