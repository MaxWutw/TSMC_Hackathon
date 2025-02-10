from utils import convert_image_to_base64, convert_base64_to_image
from PIL import Image
import requests
import numpy as np
import random
import time

text = "Give me a dog sitting on the sofa."
# %%
url = "http://127.0.0.1:8002/imagen"
payload = {
    'text': text,
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
convert_base64_to_image(image).save('output1.png')
