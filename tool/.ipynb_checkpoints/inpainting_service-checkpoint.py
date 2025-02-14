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
from utils import convert_base64_to_image, get_base64, get_file_b64str
import os
import importlib.util
import base64

app = Flask(__name__)
@app.route('/inpainting', methods=['POST'])
def generate():
    output_file = "output_img/input-image2.png"

    input_data = request.get_json()
    raw_image = input_data['img_base64']
    raw_image = convert_base64_to_image(raw_image)
    # with open("tmp_inpainting.png", "wb") as ti:
    #     ti.write(base64.decodebytes(raw_image))
    raw_image.save("tmp_inpainting.png")
    base_img = vertexImage.load_from_file(location="tmp_inpainting.png")
    text = input_data['text']
    # text = text + ".Please generate the image base on the below image(which had been turn into base64):" + raw_image
    if '.' not in text:
        text+='.'
    # model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
    model = ImageGenerationModel.from_pretrained("imagegeneration")

    images = model.edit_image(
        base_image=base_img,
        mask_mode="foreground",
        prompt=text,
        edit_mode="inpainting-remove",
    )

    images[0].save(location=output_file, include_generation_parameters=False)

    print(f"Created output image using {len(images[0]._image_bytes)} bytes")
    ret = get_file_b64str(output_file)
    return jsonify({"base64": ret}) 

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
    app.run(host=config.IP, port=config.API_PORT, threaded=False)

