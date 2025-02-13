from flask import Flask, request, jsonify
from transformers import pipeline
import numpy as np
import base64
import io
from PIL import Image
import torch
from torchvision.ops.boxes import batched_nms

device = 0 if torch.cuda.is_available() else -1
generator = pipeline("mask-generation", model="facebook/sam-vit-large", device=device)

app = Flask(__name__)

def convert_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")

def convert_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

@app.route('/sam', methods=['POST'])
def sam_auto():
    try:
        input_data = request.get_json()
        img_base64 = input_data['img_base64']

        raw_image = convert_base64_to_image(img_base64)

        results = generator(raw_image, points_per_batch=256)

        mask_images_base64 = []
        for mask in results["masks"]:
            mask_image = Image.fromarray(mask.astype("uint8") * 255)

            buffer = io.BytesIO()
            mask_image.save(buffer, format="PNG")
            mask_base64 = base64.b64encode(buffer.getvalue()).decode()
            mask_images_base64.append(mask_base64)

        return jsonify({"masks": mask_images_base64})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8002, threaded=False)
