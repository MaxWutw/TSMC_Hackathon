#%%
from utils import convert_image_to_base64
from PIL import Image
import requests
import numpy as np
import random
import time

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
url = "../va/test.png"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open(url)
text = "a cat."
# %%
# url = "http://192.168.1.100:8001/dino"
url = "http://127.0.0.1:8002/dino"
payload = {
    'text': 'cat, but you should only contains the cat without the background.',
    'img_base64': convert_image_to_base64(image)
}

headers = {'Content-Type': 'application/json'}
tic = time.time()
response = requests.post(url, json=payload, headers=headers)
print("The response:")
print(response)
toc = time.time()
print(f"DINO spend: {round(toc-tic, 3)} s")
bboxes = None
if response.status_code == 200:
    print("Response from server:", response.json())
    bboxes = response.json()
else:
    print("Failed to connect to the server")
# %%
img = np.array(image)
for bbox in bboxes:
    # payload={'image': b64_image}
    bbox = [int(i) for i in bbox]
    x1,y1,x2,y2 = bbox
    img[y1:y2, x1:x2, 2] = 255
Image.fromarray(img).save('output.png')
