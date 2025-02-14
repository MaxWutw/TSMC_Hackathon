import time
import requests
from utils import convert_image_to_base64
import PIL
import numpy as np
from thefuzz import fuzz

image = PIL.Image.open("../data/public_set/8cda5be1-f41a-49b6-afe8-f87687de4bf9/ref.png")

url = "http://127.0.0.1:8002/cloudVisionObjectDetection"

payload = {
    'img_base64': convert_image_to_base64(image)
}

headers = {
    'Content-Type': 'application/json'
}

tic = time.time()

response = requests.post(url, json=payload, headers=headers)
print("The response: ")
print(response)

toc = time.time()

print(f"Cloud Service spend: {round(toc-tic, 3)} s")

if response.status_code == 200:
    response = response.json()
    objects  = response.get("objects", [])
    
    similar_objects = []

    for object in objects:
        print(object)

        # ratio = fuzz.partial_ratio(object['name'], 'bicycle')
        ratio = fuzz.token_sort_ratio(object['name'], 'bicycle')
        print(ratio)
        
        if ratio >= 70:
            similar_objects.append(object['bounding_poly'])
else:
    print("Failed to connect to the server")
    
print(similar_objects)
