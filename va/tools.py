import api
import base64
import time
import requests
from utils import convert_image_to_base64
import importlib.util
import PIL
import numpy as np
import io

class FUNCTIONS:
    """ All available functions for the GPT calling """
    """ Format:

        Parameters:
            
        Returns:
            
        Example:
        -------
            >>> 
            
    """

    def calculate(expression: str):
        """
        'calculate' is a toolEvaluates a mathematical expression provided as a string and returns the result.
        The expression is evaluated using Python's built-in eval function, so it should be
        formatted using Python's syntax for mathematical operations.
        Parameters:
            expression (str): The mathematical expression to evaluate. This should be a valid 
                            Python expression using floating point syntax if necessary.
        Returns:
            str: The result of the evaluated expression converted to a string.
        Example:
        -------
            >>> calculate("4 * 7 / 3")
            '9.333333333333334'
        """
        return str(eval(expression))
    
    def call_grounding_dino(prompt, image):
        """'call_grounding_dino' is a tool that can detect and count multiple objects given a text
        prompt such as category names or referring expressions. The categories in text prompt
        are separated by commas or periods. It returns a list of bounding boxes with
        normalized coordinates, label names and associated probability scores.

        Parameters:
            prompt (str): The prompt to ground to the image.
            image (PIL.Image): The image to ground the prompt to.

        Returns:
            List[List[float]]: A list containing the bounding box of the detected objects (xmin, ymin, xmax, ymax). 
                xmin and ymin are the coordinates of the top-left and xmax and ymax are 
                the coordinates of the bottom-right of the bounding box.

        Example:
        -------
            >>> call_grounding_dino("cat", image)
            [[1353, 183, 1932., 736],
            [296, 64, 1147, 663]]
        """
        # url = "http://192.168.1.100:8001/dino"
        
        url = "http://127.0.0.1:8002/dino"
        # url = self.config.IP + ":" + str(self.config.API_PORT) + "/dino"
        payload = {
            'text': prompt,
            'img_base64': convert_image_to_base64(image)
        }

        headers = {'Content-Type': 'application/json'}
        tic = time.time()
        response = requests.post(url, json=payload, headers=headers)
        toc = time.time()
        print(f"DINO spend: {round(toc-tic, 3)} s")
        bboxes = None
        if response.status_code == 200:
            print("Response from server:", response.json())
            bboxes = response.json()
        else:
            print("Failed to connect to the server")
        return bboxes

    def call_inpainting_image_generate(prompt, image):
        """'call_inpainting_image_generate' is a tool that helps you replace specific targets in an image with other images. 
        You can specify which targets you want to replace using a prompt.
        When using it, you only need to provide me with the original image and a prompt specifying which part you want to replace. 
        There is no need to provide a mask.
        The return value is a JSON object containing a single field: the base64-encoded string of the modified image. 
        You can retrieve this string using return_image['base64'].

        Parameters:
            prompt (str): The prompt to ground to the image.
            image (PIL.Image): The image to ground the prompt to.

        Returns:
            Base64 encoding: A method of representing binary data using 64 printable characters.
            The returned data will be in the format: {'base64': 'encoded data'}.

        Example:
        -------
            >>> call_inpainting_image_generate("Replace the two cats in the image with two small dogs.", image)
            {'base64': 'two dogs image base64 encoded value'}
        """
        # url = self.config.IP + ":" + str(self.config.API_PORT) + "/inpainting"
        
        url = "http://127.0.0.1:8002/inpainting"
        payload = {
            'text': prompt,
            'img_base64': convert_image_to_base64(image)
        }
        headers = {'Content-Type': 'application/json'}
        tic = time.time()
        response = requests.post(url, json=payload, headers=headers)
        toc = time.time()
        print(f"Inpainting spend: {round(toc - tic, 3)} s")
        if response.status_code == 200:
           # print("Response from server:", response.json())
           gen_image = response.json()['base64']
        else:
           print("Failed to connect to the server")

        return {"base64": gen_image}

#     def call_text_to_image_generate(prompt):
#         """'call_text_to_image_generate' is a tool that generates an image based on a given prompt.
#         By providing a detailed prompt, you can obtain a result that better matches your expectations.
#         The return value is a JSON object containing a single field: the base64-encoded string of the generated image. 
#         You can retrieve this string using return_image['base64'].

#         Parameters:
#             prompt (str): The prompt to ground to the image.

#         Returns:
#             Base64 encoding: A method of representing binary data using 64 printable characters.
#             The returned data will be in the format: {'base64': 'encoded data'}.

#         Example:
#         -------
#             >>> call_text_to_image_generate("help me generate a dog sitting on the sofa")
#             {'base64': 'a dog sitting on the sofa base64 encoded value'}
#         """
#         # url = self.config.IP + ":" + str(self.config.API_PORT) + "/imagen"
        
#         url = "http://127.0.0.1:8002/imagen"
#         payload = {
#             'text': prompt,
#         }

#         headers = {'Content-Type': 'application/json'}

#         tic = time.time()
#         response = requests.post(url, json=payload, headers=headers)
#         toc = time.time()
#         print(f"Imagen spend: {round(toc - tic, 3)} s")
#         if response.status_code == 200:
#            # print("Response from server:", response.json())
#            gen_image = response.json()['base64']
#         else:
#            print("Failed to connect to the server")

#         return {"base64": gen_image}
    
    def call_segment_anything_model(image):
        """'call_vit_sam' is a tool that helps you generate high-quality object segmentation masks from an input image.
        You can let the model automatically generate masks by prividing the original image.
        The tool interacts with a SAM API server to perform segmentation and returns a base64-encoded image with RED blended masks.

        When using it, you only need to provide the original image. There is only one parameter need to pass in.
        The model processes the image and returns segmentation masks, which are blended with the original image.
        The return value is a JSON object containing a single field: the base64-encoded string of the modified image.
        You can retrieve this string using return_image['base64'].
        Please note that the output sets the masked areas to 255, which means they will appear red.

        Parameters:
            image (PIL.Image): The image to ground the prompt to.

        Returns:
            Base64 encoding: A method of representing binary data using 64 printable characters.
            The returned data will be in the format: {'base64': 'encoded data'}.
        Example:
        -------
            >>> call_segment_anything_model(image)
            {'base64': 'segmented image base64 encoded value'}
        """

        url = "http://127.0.0.1:8002/sam"
        payload = {
            "img_base64": convert_image_to_base64(image)
            # "img_base64": image
        }

        headers = {'Content-Type': 'application/json'}

        tic = time.time()
        response = requests.post(url, json=payload, headers=headers)
        toc = time.time()

        print(f"SAM API response time: {round(toc - tic, 3)} s")

        if response.status_code == 200:
            response_json = response.json()
            masks_base64 = response_json.get("base64", [])

            if not masks_base64:
                print("No masks returned from the server.")
                exit()

            print(f"Received {len(masks_base64)} masks")

            img_np = np.array(image)

            for mask_base64 in masks_base64:
                mask_data = base64.b64decode(mask_base64)
                mask_image = PIL.Image.open(io.BytesIO(mask_data)).convert("L") 
                mask_np = np.array(mask_image)

                img_np[mask_np > 128, 0] = 255

            output_image = PIL.Image.fromarray(img_np)
            output_image.save("sam_middle_image.png")
            print("Saved blended image as output.png")
        else:
            print(f"Failed to connect to the server, status code: {response.status_code}")
            print("Response:", response.text)

        gen_image = convert_image_to_base64(output_image)

        return {"base64": gen_image}

    def call_cloud_vision_object_detect(image):
        """
        'call_cloud_vision_object_detect' is an object detection function that processes an image to identify multiple objects 
    and returns their bounding polygons along with relevant details.

        Parameters:
            image (PIL.Image): The image to ground the prompt to.
        
        Returns:
            List[Dict[str, Any]]: A list of detected objects, where each object contains:
            - "name" (str): The label of the detected object.
            - "score" (float): The confidence score of the detection.
            - "bounding_poly" (List[List[float]]): The normalized bounding polygon of the detected object.
        
        Example:
        ---------------
        >>> call_cloud_vision_object_detect(image)
        [
            {
                "name": "Cat",
                "score": 0.95,
                "bounding_poly": [[0.23, 0.15], [0.75, 0.45], [0.30, 0.55]]
            },
            {
                "name": "Dog",
                "score": 0.89,
                "bounding_poly": [[0.05, 0.10], [0.90, 0.80]]
            }
        ]
        """
        # url = self.config.IP + ":" + str(self.config.API_PORT) + "/cloud_vision_object_detect"
        
        url = "http://127.0.0.1:8002/cloudVisionObjectDetection"

        payload = {
            'img_base64': convert_image_to_base64(image)
        }
        headers = {'Content-Type': 'application/json'}
        tic = time.time()
        response = requests.post(url, json=payload, headers=headers)
        toc = time.time()
        print(f"cloud vision object detect spend: {round(toc - tic, 3)} s")
        if response.status_code == 200:
            print("Response from server:", response.json())
            # gen_image = response.json()['base64']
            response_json = response.json()
            detected_objects = response_json.get("objects", [])
            
            # similar_objects = []
            
            # for object in detected_objects:
            #     print(object['bounding_poly'])
            #     similar_objects.append(object['bounding_poly'])
            
            return detected_objects
            
        else:
            print("Failed to connect to the server")
            return []

#         return similar_objects
