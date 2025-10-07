# utils/faiss_utils.py
import numpy as np
import time
from typing import List, Dict
from vector_db.faiss import FAISSHNSWStore
from embedding.embedder import Embedders


class FAISSUtils:
    def __init__(self, dimension: int = 768, M: int = 32,
                 ef_construction: int = 200, ef_search: int = 50):
        """
        Initialize FAISS utils with HNSW parameters
        """
        self.dimension = dimension
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.embedder = Embedders()

    def build_faiss_index(self, embedded_chunks: List[Dict], index_name: str = "my_knowledge_base"):
        """
        Build and save FAISS index from embedded chunks
        """
        print("\nInitializing FAISS HNSW store...")
        faiss_store = FAISSHNSWStore(
            dimension=self.dimension,
            M=self.M,
            ef_construction=self.ef_construction,
            ef_search=self.ef_search
        )

        # Prepare data for FAISS
        embeddings_list = []
        metadata_list = []
        chunk_texts = []

        for chunk in embedded_chunks:
            embeddings_list.append(chunk['embedding'])
            metadata_list.append(chunk['metadata'])
            chunk_texts.append(chunk['text'])

        # Convert to numpy array for FAISS
        embeddings_array = np.array(embeddings_list).astype('float32')

        # Add embeddings to FAISS
        faiss_store.add_embeddings(
            embeddings_array, metadata_list, chunk_texts)

        # Print FAISS statistics
        stats = faiss_store.get_index_stats()
        print("\nFAISS Index Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Save the FAISS index to vector_db folder
        faiss_store.save_index(index_name)

        return faiss_store

    def load_faiss_index(self, index_name: str = "my_knowledge_base"):
        """
        Load existing FAISS index from vector_db folder
        """
        faiss_store = FAISSHNSWStore()
        faiss_store.load_index(index_name)
        return faiss_store

    def  test_search_functionality(self, faiss_store: FAISSHNSWStore, test_queries: List[str] = None):
        """
        Test search functionality with sample queries
        """
        if test_queries is None:
            test_queries = [
                "JavaScript interview questions",
                "What are closures in programming?",
                "Explain event loop in JavaScript"
            ]

        print("\nTesting search functionality...")

        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: '{query}'")
            print(f"{'='*60}")

            # Embed the query
            query_embedding = self.embedder.Embedding(
                [query])[0].astype('float32')
            query_embedding = np.expand_dims(query_embedding, axis=0)

            # Search in FAISS
            results = faiss_store.search_with_threshold(query_embedding, k=3)

            for result in results:
                print(f"\nRank {result['rank']}")
                print(f"Distance: {result['distance']:.4f}")
                print(f"Similarity Score: {result['score']:.4f}")
                print(f"Text: {result['text'][:100]}...")
                print(
                    f"Source: {result['metadata']['filename']} (Page {result['metadata']['page_number']})")
                print(f"Search time: {result.get('search_time_ms', 0):.2f}ms")
