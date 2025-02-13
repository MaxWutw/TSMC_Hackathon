import pytest
import requests
import time
from utils import convert_image_to_base64, convert_base64_to_image
from PIL import Image

BASE_URL = "http://127.0.0.1:8002"

@pytest.fixture
def test_image():
    image_path = "ref.png"
    image = Image.open(image_path)
    return convert_image_to_base64(image)

def test_inpainting(test_image):
    url = f"{BASE_URL}/inpainting"
    text = "Help me convert the bear in the images into lion."

    payload = {
        'text': text,
        'img_base64': test_image
    }

    headers = {'Content-Type': 'application/json'}

    tic = time.time()
    response = requests.post(url, json=payload, headers=headers)
    toc = time.time()
    elapsed_time = round(toc - tic, 3)

    assert response.status_code == 200, f"API request failed with status {response.status_code}"
    
    json_data = response.json()
    assert 'base64' in json_data, "Response JSON does not contain 'base64' key"
    
    output_image = convert_base64_to_image(json_data['base64'])
    output_image.save("output.png")

    print(f"Test passed! API response time: {elapsed_time} s")
