#%%
from utils import convert_image_to_base64, convert_base64_to_image
from PIL import Image
import requests
import numpy as np
import random
import time

url = "../va/ref.png"
image = Image.open(url)
text = "Help me convert the bear in the images into lion.,You can detect the bear in the image and use image generator to generate the lino and replace the bear to lion.Supply the end result as a PIL.Image object."
# %%
url = "http://127.0.0.1:8002/inpainting"
payload = {
    'text': text,
    'img_base64': convert_image_to_base64(image)
}

headers = {'Content-Type': 'application/json'}
tic = time.time()
response = requests.post(url, json=payload, headers=headers)
print("The response:")
print(response)
toc = time.time()
print(f"Imagen spend: {round(toc - tic, 3)} s")
if response.status_code == 200:
    # print("Response from server:", response.json()['base64'])
    image = response.json()['base64']
else:
    print("Failed to connect to the server")
# %%
convert_base64_to_image(image).save('output.png')
