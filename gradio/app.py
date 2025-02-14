import gradio as gr
from PIL import Image
from io import BytesIO
import requests
import base64
import numpy as np

url = "http://34.30.44.80:8001/vision"

def user_interface():
    
    def send_request(payload):
    # with open("test_input.json", "r") as file:
    #     payload = json.load(file)
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=payload, headers=headers)
        print(response)
        if response.status_code == 200:
            # print("Response from server:", response.json())
            image = response.json()['result']
        else:
            print("Failed to connect to the server")
            image = None

        return image

    def get_payload(a , b , c , d):
        payload = {
            "groupToken": "XXX",
            "uuid": "0",
            "messages": "",
            "detail": b,
            "outputType": a,
            "artifacts": []
        }
        print(payload)
        payload["artifacts"] = get_artifact(a,b,c,d)

        return payload


    def get_artifact(a,b,c,d):
        
        
        try:
            ret = [
            {
                "base64": base64.b64encode(c).decode("utf-8"),
                "type": 'png',
                "name": 'ref'
            },
            {
                "base64": base64.b64encode(d).decode("utf-8"),
                "type": 'png',
                "name": 'test'
            }
            ]
        except:
            ret = [
             {
                "base64": base64.b64encode(c).decode("utf-8"),
                "type": 'png',
                "name": 'ref'
             }
            ]
        
        return ret

    def output(a,b,c,d):
        
        result = send_request(get_payload(a,b,c,d))
        print(result,type(result))
        
        try:
            return None, eval(result)
        except:
            try:
                return Image.open(BytesIO(base64.b64decode(result))),None
            except:
                return None,None
    
    input_ = [
        gr.Text( 
            label="Output Format"
        ),
        gr.Text(label="Detail"),
        gr.Image(height = 200, label="Ref"),
        gr.Image(height = 200, label="Test")
    ]

    output_ = [
        gr.Image(height = 200, label="Image"),
        gr.Text(label="Int/Float (Number)")
    ]

    with gr.Blocks() as app:
        gr.Interface(
            title="TSMC Career Hackathon A2 Fall in TSMC",
            fn=output,
            inputs=input_,
            outputs=output_,
            css="footer{display:none !important}"
        )

    return app

demo = user_interface()
demo.launch(server_name="0.0.0.0" , server_port=8005)