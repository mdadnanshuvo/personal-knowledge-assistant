# utils/embedding_utils.py
import json
from typing import List, Dict
from embedding.embedder import Embedders

def embed_chunks_with_metadata(chunks_with_metadata: List[Dict]) -> List[Dict]:
    """
    Embed chunks with metadata using the Embedder class
    """
    embedder = Embedders()
    print(f"\nEmbedding {len(chunks_with_metadata)} chunks...")
    embedded_chunks = embedder.embed_with_metadata(chunks_with_metadata)
    return embedded_chunks

