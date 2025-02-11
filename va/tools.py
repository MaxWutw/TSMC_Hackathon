import api
import time
import requests
from utils import convert_image_to_base64
import importlib.util
import PIL

class FUNCTIONS:
    """ All available functions for the GPT calling """
    """ Format:

        Parameters:
            
        Returns:
            
        Example:
        -------
            >>> 
            
    """

    def call_google(query: str, **kwargs):
        """ 
        'call_google' returns a summary from searching query on google
        Parameters:
            query (str): The search query to be used in the Google search.
            **kwargs: Additional keyword arguments that can be used for extended functionality
                  in future implementations, but are currently unused.
        Returns:
            str: A response message searched by google.
        Example
        -------
            >>> call_google(European Union)
            "Sweden decided to join NATO due to a combin..."
        """
        mock_info = "Sweden decided to join NATO due to a combination of reasons. These include the desire to strengthen its security in light of recent regional developments, notably Russia's aggressive actions, \
                    such as its war that has pushed the alliance to expand. Sweden, along with Finland, has been an official partner of NATO since 1994 and has been a major contributor to the alliance, participating in various NATO operations. \
                    The Swedish government assessed that joining NATO was the best way to protect its national security. Additionally, Sweden's membership was seen as a means to strengthen NATO itself and add to the stability in the Euro-Atlantic area. \
                    The process of joining NATO was completed after overcoming the last hurdles, including obtaining the necessary approvals from all existing member states, such as Turkey and Hungary."
        mock_info = 'Cannot find result, you can search on wikipedia.'
        return mock_info

    def call_wikipedia(query: str):
        """ 
        'call_wikipedia' returns a summary from searching query on wiki
        Parameters:
            query (str): The search query to be used in the wiki search.
        Returns:
            str: A response message searched on wiki.
        Example
        -------
            >>> call_wikipedia(European Union)
            "Sweden decided to join NATO due to a combin..."
        """
        mock_info = "Sweden decided to join NATO due to a combination of reasons. These include the desire to strengthen its security in light of recent regional developments, notably Russia's aggressive actions, \
                    such as its war that has pushed the alliance to expand. Sweden, along with Finland, has been an official partner of NATO since 1994 and has been a major contributor to the alliance, participating in various NATO operations. \
                    The Swedish government assessed that joining NATO was the best way to protect its national security. Additionally, Sweden's membership was seen as a means to strengthen NATO itself and add to the stability in the Euro-Atlantic area. \
                    The process of joining NATO was completed after overcoming the last hurdles, including obtaining the necessary approvals from all existing member states, such as Turkey and Hungary."
        
        return mock_info


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
        You can retrieve this string using return_image.json()['base64'].

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

        return gen_image

    def call_text_to_image_generate(prompt):
        """'call_text_to_image_generate' is a tool that generates an image based on a given prompt.
        By providing a detailed prompt, you can obtain a result that better matches your expectations.
        The return value is a JSON object containing a single field: the base64-encoded string of the generated image. 
        You can retrieve this string using return_image.json()['base64'].

        Parameters:
            prompt (str): The prompt to ground to the image.

        Returns:
            Base64 encoding: A method of representing binary data using 64 printable characters.
            The returned data will be in the format: {'base64': 'encoded data'}.

        Example:
        -------
            >>> call_inpainting_image_generate("help me generate a dog sitting on the sofa")
            {'base64': 'a dog sitting on the sofa base64 encoded value'}
        """
        # url = self.config.IP + ":" + str(self.config.API_PORT) + "/imagen"
        
        url = "http://127.0.0.1:8002/imagen"
        payload = {
            'text': prompt,
        }

        headers = {'Content-Type': 'application/json'}
        tic = time.time()
        response = requests.post(url, json=payload, headers=headers)
        toc = time.time()
        print(f"Imagen spend: {round(toc - tic, 3)} s")
        if response.status_code == 200:
           # print("Response from server:", response.json())
           gen_image = response.json()['base64']
        else:
           print("Failed to connect to the server")

        return gen_image
