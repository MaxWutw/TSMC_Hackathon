import base64
import requests
import numpy as np
import time
from PIL import Image
import io

def convert_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

image_base64 = convert_image_to_base64(raw_image)

url = "http://127.0.0.1:8002/sam"
payload = {"img_base64": image_base64}

headers = {'Content-Type': 'application/json'}

tic = time.time()
response = requests.post(url, json=payload, headers=headers)
toc = time.time()

print(f"SAM API response time: {round(toc - tic, 3)} s")

if response.status_code == 200:
    response_json = response.json()
    masks_base64 = response_json.get("base64", [])

    if not masks_base64:
        print("No masks returned from the server.")
        exit()

    print(f"Received {len(masks_base64)} masks")

    img_np = np.array(raw_image)

    for mask_base64 in masks_base64:
        mask_data = base64.b64decode(mask_base64)
        mask_image = Image.open(io.BytesIO(mask_data)).convert("L") 
        mask_np = np.array(mask_image)

        img_np[mask_np > 128, 0] = 255  

    output_image = Image.fromarray(img_np)
    output_image.save("output.png")
    print("Saved blended image as output.png")
else:
    print(f"Failed to connect to the server, status code: {response.status_code}")
    print("Response:", response.text)
