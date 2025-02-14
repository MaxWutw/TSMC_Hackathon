import json
import requests
from concurrent.futures import ThreadPoolExecutor
from utils import convert_image_to_base64, convert_base64_to_image
import pandas as pd
import os
from PIL import Image
import sys
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the module
module_dir = os.path.abspath(os.path.join(current_dir, '../'))

# Add the module directory to sys.path
sys.path.append(module_dir)
import utils



# url = "http://192.168.1.100:8001/vision"
url = "http://127.0.0.1:8001/vision"

def send_request(payload):
    # with open("test_input.json", "r") as file:
    #     payload = json.load(file)
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=payload, headers=headers)
    print(response)
    if response.status_code == 200:
        # print("Response from server:", response.json())
        result = response.json()['result']
        if isinstance(result, Image.Image):
            print(f"The result is an image: inpainting_image.png")
            convert_base64_to_image(result).save('inpainting_image.png')
        elif isinstance(result, (int, float)):
            result = float(result)
            print(f"The result is a number: {result}")
    else:
        print("Failed to connect to the server")

def get_payload(ob, data_root):
    payload = {
        "groupToken": "XXX",
        "uuid": ob['id'],
        "messages": ob['messages'],
        "detail": ob['detail'],
        "outputType": ob['output_format'],
        "artifacts": []
    }
    print(payload)
    payload["artifacts"] = get_artifact(ob, data_root)
    return payload


def get_artifact(ob, data_root):
    ret = []
    input_dict = eval(ob['input'])
    for k in input_dict:
        obj = {
            "base64": utils.convert_image_to_base64(Image.open(os.path.join(data_root, input_dict[k]))),
            "type": 'png',
            "name": k
        }
        ret.append(obj)
    return ret


if __name__ == '__main__':
    num_requests = 1
    csv_path = f'../data/testcase.csv'
    # csv_path = f'../data/release_public_set.csv'
    data_root = ''
    output_root = '../test_output'
    db = pd.read_csv(csv_path)
    ob = db.iloc[0,:]
    send_request(get_payload(ob, data_root))
    # with ThreadPoolExecutor(max_workers=num_requests) as executor:
    #     futures = [executor.submit(send_request) for _ in range(num_requests)]
    #     for future in futures:
    #         future.result()
