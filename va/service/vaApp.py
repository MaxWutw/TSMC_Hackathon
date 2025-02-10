import os
import sys
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the module
module_dir = os.path.abspath(os.path.join(current_dir, '../'))

# Add the module directory to sys.path
sys.path.append(module_dir)

from flask import Flask, jsonify, request
import json
import time
import pandas as pd
from PIL import Image
from utils import check_dir, convert_image_to_base64, parser_np
from data_manager import DataManagerService
from agent.vision_assistant_agent import VisionAssistant
import importlib.util

app = Flask(__name__)
@app.route('/vision', methods=['POST'])
def process_data():
    input_data = request.get_json()
    image_data = input_data["artifacts"]
    prompt = input_data["messages"]
    detail_prompt = input_data["detail"]

    output_type = input_data["outputType"]
    uuid = input_data["uuid"]

    df = pd.DataFrame({"messages": [prompt], "detail": [detail_prompt], "input": [json.dumps(image_data)]})
    db = DataManagerService(df)
    messages, images, ob = db[0]
    va = VisionAssistant(debug=False, timeout=120, memory_limit_mb=200)
    result = va.predict(messages, images)

    if isinstance(result, Image.Image):
        result = convert_image_to_base64(result)
    elif isinstance(result, (int, float)):
        result = float(result)
    
    result = parser_np({'result': result})
    
    # va.dump_record()
    return jsonify({
        "result": result['result'],
        "type": output_type,
        "uuid": uuid
    })

@app.route('/api', methods=['GET'])
def hello_world():
    return jsonify(message="Hello, World!")

if __name__ == '__main__':
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "config.py"))
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # app.run(host=config.IP, port=config.VISIONASSISTANT_PORT, threaded=True)
    app.run(host="0.0.0.0", port=8001, threaded=True)
