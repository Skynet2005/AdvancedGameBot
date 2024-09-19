"""
/api/ollama_helper.py

This module provides a helper class for interacting with the Ollama API. It includes functionalities to generate text based on a given prompt and to derive insights from vision analysis data.
"""

import requests
import json

class OllamaHelper:
    def __init__(self, base_url="http://localhost:11434"):
        # Initialize the base URL for the Ollama API
        self.base_url = base_url

    def generate(self, prompt, model="llama:3.1", max_tokens=100):
        # Construct the URL for the generate endpoint
        url = f"{self.base_url}/api/generate"
        
        # Prepare the data payload for the API request
        data = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens
        }

        # Make a POST request to the Ollama API
        response = requests.post(url, json=data)
        if response.status_code == 200:
            # Return the response if the request was successful
            return response.json()['response']
        else:
            # Raise an exception if there was an error in the API call
            raise Exception(f"Error in Ollama API call: {response.status_code} - {response.text}")

    def get_llama_insights(self, vision_analysis):
        # Create a prompt based on the vision analysis
        prompt = f"""
        Based on the following vision analysis of a game state, provide insights and suggestions:

        Objects detected: {', '.join([obj[0] for obj in vision_analysis['objects']])}
        Scene: {vision_analysis['scene'][0][0]}
        Segmentation summary: {self._summarize_segmentation(vision_analysis['segmentation'])}

        What actions or strategies would you recommend for this game state?
        """

        # Generate insights using the Ollama API
        return self.generate(prompt)

    # TODO: Implement a more sophisticated summary based on your specific segmentation output.
    def _summarize_segmentation(self, segmentation):
        # Summarize the segmentation data
        unique, counts = np.unique(segmentation, return_counts=True)
        return dict(zip(unique, counts))

# Initialize the OllamaHelper instance
ollama_helper = OllamaHelper()

def get_llama_insights(vision_analysis):
    # Retrieve insights using the OllamaHelper instance
    return ollama_helper.get_llama_insights(vision_analysis)