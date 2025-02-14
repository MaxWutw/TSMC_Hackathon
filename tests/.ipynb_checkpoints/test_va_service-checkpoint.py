import pytest
import requests
import os
import pandas as pd
from utils import convert_image_to_base64, convert_base64_to_image
from PIL import Image

BASE_URL = "http://127.0.0.1:8003"

@pytest.fixture
def test_data():
    csv_path = os.path.abspath("../testcase.csv")
    db = pd.read_csv(csv_path)
    data_root = ""
    return db.iloc[0, :], data_root

def get_payload(ob, data_root):
    payload = {
        "groupToken": "XXX",
        "uuid": ob["id"],
        "messages": ob["messages"],
        "detail": ob["detail"],
        "outputType": ob["output_format"],
        "artifacts": []
    }
    payload["artifacts"] = get_artifact(ob, data_root)
    return payload

def get_artifact(ob, data_root):
    ret = []
    input_dict = eval(ob["input"])
    for k in input_dict:
        img_path = os.path.join(data_root, input_dict[k])
        obj = {
            "base64": convert_image_to_base64(Image.open(img_path)),
            "type": "png",
            "name": k
        }
        ret.append(obj)
    return ret

def test_vision_api(test_data):
    ob, data_root = test_data
    url = f"{BASE_URL}/vision"
    headers = {"Content-Type": "application/json"}
    payload = get_payload(ob, data_root)

    response = requests.post(url, json=payload, headers=headers)
    
    assert response.status_code == 200, f"API request failed with status {response.status_code}"

    json_data = response.json()
    assert "result" in json_data, "Response JSON does not contain 'result' key"

    output_image = convert_base64_to_image(json_data["result"])
    output_image.save("test_output.png")

    print("Test passed! Image saved as test_output.png")

