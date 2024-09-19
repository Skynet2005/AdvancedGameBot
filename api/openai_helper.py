"""
/api/openai_helper.py

This module provides helper functions for interacting with the OpenAI API to 
generate action suggestions based on the current game state and relevant documents 
retrieved from a knowledge base.

"""

import os
import openai
from dotenv import load_dotenv
from .retriever import DocumentRetriever
import logging

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


class LlamaRAGHelper:
    def __init__(self, api_key, retriever: DocumentRetriever):
        self.api_key = api_key
        self.retriever = retriever

    def get_goal(self, game_screenshot, query):
        try:
            # Retrieve relevant documents based on the query
            relevant_docs = self.retriever.retrieve(query)
            # Augment the query with the retrieved documents
            augmented_query = f"{query}\n\nHere are some relevant documents: \n\n" + \
                "\n".join(relevant_docs)

            # Create the prompt for the OpenAI API
            prompt = (
                f"Given the game state described in this screenshot: {game_screenshot.shape}, "
                f"and this query: '{augmented_query}', provide action suggestions."
            )
            # Use the OpenAI ChatCompletion API with streaming
            response = openai.ChatCompletion.create(
                model="gpt-4-mini",  # Use your desired model
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            suggestion = ""
            # Process the streaming response
            for chunk in response:
                if 'choices' in chunk:
                    delta = chunk['choices'][0].get('delta', {})
                    content = delta.get('content', '')
                    suggestion += content
            return suggestion.strip()
        except openai.error.OpenAIError as e:
            # Log OpenAI API errors
            logging.error(f"OpenAI API error: {str(e)}")
            return None
        except Exception as e:
            # Log unexpected errors
            logging.error(f"Unexpected error in get_game_goal: {str(e)}")
            return None


def get_goal_suggestion(game_state, query):
    # Initialize the LlamaRAGHelper with necessary parameters
    # Note: You need to properly initialize the DocumentRetriever
    retriever = DocumentRetriever("path/to/your/document/store")
    rag_helper = LlamaRAGHelper(openai.api_key, retriever)

    # Call the get_goal method and return the result
    return rag_helper.get_goal(game_state, query)
