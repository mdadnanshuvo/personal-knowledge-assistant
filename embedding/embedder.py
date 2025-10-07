from FlagEmbedding import FlagModel
import numpy as np

class Embedders:
    def __init__(self):
        self.model = FlagModel("BAAI/bge-base-en-v1.5", use_fp16=True)

    def Embedding(self, texts):
        """Embed text strings"""
        embeddings = self.model.encode(texts)
        return embeddings

    def embed_with_metadata(self, chunks_with_metadata):
        """
        Embed chunks along with their metadata
        chunks_with_metadata: list of dicts with 'text' and 'metadata' keys
        Returns: list of dicts with embeddings and metadata
        """
        texts = [item['text'] for item in chunks_with_metadata]
        embeddings = self.model.encode(texts)
        
        results = []
        for i, embedding in enumerate(embeddings):
            result = {
                'embedding': embedding.tolist(),  # Convert numpy array to list
                'text': chunks_with_metadata[i]['text'],
                'metadata': chunks_with_metadata[i]['metadata']
            }
            results.append(result)
        
        return results