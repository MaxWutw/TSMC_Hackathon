import pytest
import requests
import time
from utils import convert_base64_to_image

BASE_URL = "http://127.0.0.1:8002"

@pytest.fixture
def test_text():
    return "Give me a dog sitting on the sofa."

def test_imagen(test_text):
    url = f"{BASE_URL}/imagen"
    payload = {'text': test_text}
    headers = {'Content-Type': 'application/json'}

    tic = time.time()
    response = requests.post(url, json=payload, headers=headers)
    toc = time.time()
    elapsed_time = round(toc - tic, 3)

    assert response.status_code == 200, f"API request failed with status {response.status_code}"

    json_data = response.json()
    assert 'base64' in json_data, "Response JSON does not contain 'base64' key"

    output_image = convert_base64_to_image(json_data['base64'])
    output_image.save

