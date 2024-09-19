"""
/utils/rag_utils.py

This file provides utility functions for a Retrieval-Augmented Generation (RAG) system. 
It includes functionalities to load a knowledge base, build a FAISS index for efficient 
similarity search, and retrieve relevant information based on the current game state. 
Additionally, it allows adding new information to the knowledge base and updating the FAISS index.

"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

class RAGUtils:
    def __init__(self, knowledge_base_path):
        self.knowledge_base_path = knowledge_base_path
        # Initialize the sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # Load the knowledge base from the specified path
        self.knowledge_base = self.load_knowledge_base()
        # Build the FAISS index from the loaded knowledge base
        self.index = self.build_index()

    def load_knowledge_base(self):
        # Check if the knowledge base file exists
        if not os.path.exists(self.knowledge_base_path):
            print(f"Knowledge base file not found at {self.knowledge_base_path}. Starting with an empty knowledge base.")
            return []
        # Read the knowledge base file and return the lines as a list
        with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def build_index(self):
        # Check if the knowledge base is empty
        if not self.knowledge_base:
            print("Knowledge base is empty. FAISS index will not be created.")
            return None
        # Encode the knowledge base entries into embeddings
        embeddings = self.model.encode(self.knowledge_base)
        # Check if any valid embeddings were created
        if embeddings.size == 0:
            print("No valid embeddings were created. FAISS index will not be created.")
            return None
        # Create a FAISS index and add the embeddings
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        return index

    def retrieve_relevant_info(self, game_state, k=5):
        # Check if the FAISS index is available
        if self.index is None:
            return []
        # Encode the game state into an embedding
        query_embedding = self.model.encode([str(game_state)])
        # Search the FAISS index for the top-k nearest neighbors
        _, indices = self.index.search(query_embedding.astype('float32'), k)
        # Retrieve the relevant information from the knowledge base
        relevant_info = [self.knowledge_base[i] for i in indices[0] if i < len(self.knowledge_base)]
        return relevant_info

    def add_new_information(self, new_info):
        # Append the new information to the knowledge base
        self.knowledge_base.append(new_info)
        # Update the FAISS index with the new embedding
        new_embedding = self.model.encode([new_info])
        if self.index is None:
            self.index = faiss.IndexFlatL2(new_embedding.shape[1])
        self.index.add(new_embedding.astype('float32'))
        # Save the updated knowledge base to file
        with open(self.knowledge_base_path, 'a', encoding='utf-8') as f:
            f.write(new_info + '\n')

# Initialize the RAGUtils instance with the path to the knowledge base
rag_utils = RAGUtils('data/game_knowledge_base.txt')

def retrieve_relevant_info(game_state):
    # Retrieve relevant information based on the game state
    return rag_utils.retrieve_relevant_info(game_state)

def add_new_information(new_info):
    # Add new information to the knowledge base
    rag_utils.add_new_information(new_info)
