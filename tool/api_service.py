import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from vertexai.preview.vision_models import Image as vertexImage
from transformers import AutoProcessor, GroundingDinoForObjectDetection
from transformers import pipeline
from PIL import Image
import requests
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from flask import Flask, jsonify, request
from utils import convert_base64_to_image, get_base64, get_file_b64str
import os
import io
import importlib.util
import base64
from torchvision.ops.boxes import batched_nms

app = Flask(__name__)
@app.route('/imagen', methods=['POST'])
def image_predict():
    input_data = request.get_json()
    output_file = "output_img/input-image1.png"
    text = input_data['text']
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
    ret = get_file_b64str(output_file)
    return jsonify({"base64": ret}) 

@app.route('/inpainting', methods=['POST'])
def inpainting_predict():
    input_data = request.get_json()
    output_file = "output_img/input-image2.png"
    print(output_file)
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

# @app.route('/dino', methods=['POST'])
# def dino_predict():
# 
#     input_data = request.get_json()
#     raw_image = input_data['img_base64']
#     raw_image = convert_base64_to_image(raw_image)
#     text = input_data['text']
#     if '.' not in text:
#         text+='.'
#     inputs = processor_dino(images=raw_image, text=text, return_tensors="pt").to(device)
#     outputs = model_dino(**inputs)
#     target_sizes = torch.tensor([raw_image.size[::-1]])
#     results = processor_dino.image_processor.post_process_object_detection(
#         outputs, threshold=0.2, target_sizes=target_sizes
#     )[0]
#     bbox_list = []
#     for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#         box = [round(i, 1) for i in box.tolist()]
#         bbox_list.append(box)
#         print(f"Detected {label.item()} with confidence " f"{round(score.item(), 2)} at location {box}")
#     return bbox_list

# @app.route('/sam', methods=['POST'])
# def sam_auto():
#     input_data = request.get_json()
#     img_base64 = input_data['img_base64']
# 
#     raw_image = convert_base64_to_image(img_base64)
# 
#     results = generator(raw_image, points_per_batch=256)
# 
#     mask_images_base64 = []
#     for mask in results["masks"]:
#         mask_image = Image.fromarray(mask.astype("uint8") * 255)
# 
#         buffer = io.BytesIO()
#         mask_image.save(buffer, format="PNG")
#         mask_base64 = base64.b64encode(buffer.getvalue()).decode()
#         mask_images_base64.append(mask_base64)
# 
#     return jsonify({"masks": mask_images_base64})

@app.route('/api', methods=['GET'])
def check():
    return jsonify(message="Imagen is running!")

if __name__ == '__main__':

    if not os.path.exists("output_img"):
        os.mkdir("output_img")

    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.py"))
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    PROJECT_ID = "tsmccareerhack2025-aaid-grp2"
    credential_path = "../va/tsmccareerhack2025.json"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(f"Your device is: {device}")
    # processor_dino = AutoProcessor.from_pretrained("grounding-dino-base")
    # model_dino = GroundingDinoForObjectDetection.from_pretrained("grounding-dino-base").to(device)


    # generator = pipeline("mask-generation", model="facebook/sam-vit-large", device=device)

    vertexai.init(project=PROJECT_ID, location="us-central1")
    app.run(host=config.IP, port=config.API_PORT, threaded=False)
