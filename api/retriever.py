"""
/api/retriever.py

This module provides a DocumentRetriever class that uses Dense Passage Retrieval (DPR) models to encode queries and documents. 
It retrieves the most relevant documents from a document store based on the similarity between query and document embeddings.
"""

import os
import torch
from transformers import DPRQuestionEncoderTokenizer, DPRQuestionEncoder, DPRContextEncoderTokenizer, DPRContextEncoder

class DocumentRetriever:
    def __init__(self, document_store):
        self.document_store = document_store
        # Initialize the question encoder and tokenizer
        self.q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        # Initialize the context encoder and tokenizer
        self.ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    
    def retrieve(self, query, top_k=5):
        # Get the embedding for the query
        query_embedding = self.get_query_embedding(query)
        # Search the document store for the top_k relevant documents
        retrieved_docs = self.search_document_store(query_embedding, top_k=top_k)
        return retrieved_docs
    
    def get_query_embedding(self, query):
        # Tokenize the query and get its embedding
        inputs = self.q_tokenizer(query, return_tensors="pt")
        return self.q_encoder(**inputs).pooler_output
    
    def search_document_store(self, query_embedding, top_k=5):
        scores = {}
        # Calculate similarity scores between the query embedding and each document embedding
        for doc_id, doc in self.document_store.items():
            doc_embedding = self.get_document_embedding(doc)
            scores[doc_id] = torch.cosine_similarity(query_embedding, doc_embedding.unsqueeze(0)).item()
        # Sort documents by their similarity scores in descending order and select the top_k
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [self.document_store[doc_id] for doc_id, score in sorted_docs]
    
    def get_document_embedding(self, doc_text):
        # Tokenize the document text and get its embedding
        inputs = self.ctx_tokenizer(doc_text, return_tensors="pt")
        return self.ctx_encoder(**inputs).pooler_output.squeeze(0)
