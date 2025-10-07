# vector_db/faiss.py

import faiss
import numpy as np
import json
import time
import os

class FAISSHNSWStore:
    def __init__(self, dimension=768, M=32, ef_construction=200, ef_search=50):
        self.dimension = dimension
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        self.index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_L2) #pylint: disable=no-member
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search
        
        self.metadata = []
        self.chunk_texts = []
        self.is_trained = False

    def add_embeddings(self, embeddings, metadata_list, chunk_texts):
        if len(embeddings) == 0:
            return
        if not self.is_trained:
            self.index.add(embeddings)
            self.is_trained = True
        else:
            self.index.add(embeddings)

        self.metadata.extend(metadata_list)
        self.chunk_texts.extend(chunk_texts)

        print(f"Added {len(embeddings)} embeddings to HNSW index")
        print(f"Total vectors in index: {self.index.ntotal}")

    def search(self, query_embedding, k=5):
        if self.index.ntotal == 0:
            return []
        original_ef_search = self.index.hnsw.efSearch
        self.index.hnsw.efSearch = max(self.ef_search, k * 2)

        start_time = time.time()
        distances, indices = self.index.search(query_embedding, k)
        search_time = time.time() - start_time

        self.index.hnsw.efSearch = original_ef_search

        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1 and idx < len(self.metadata):
                similarity_score = 1.0 / (1.0 + distance)
                results.append({
                    'rank': i + 1,
                    'distance': float(distance),
                    'score': float(similarity_score),
                    'text': self.chunk_texts[idx],
                    'metadata': self.metadata[idx],
                    'search_time_ms': search_time * 1000
                })
        return results

    def search_with_threshold(self, query_embedding, k=5, max_distance=10.0):
        results = self.search(query_embedding, k)
        return [r for r in results if r['distance'] <= max_distance]

    def batch_search(self, query_embeddings, k=5):
        if self.index.ntotal == 0:
            return []
        original_ef_search = self.index.hnsw.efSearch
        self.index.hnsw.efSearch = max(self.ef_search, k * 2)
        distances, indices = self.index.search(query_embeddings, k)
        self.index.hnsw.efSearch = original_ef_search

        all_results = []
        for query_idx, (query_distances, query_indices) in enumerate(zip(distances, indices)):
            query_results = []
            for i, (distance, idx) in enumerate(zip(query_distances, query_indices)):
                if idx != -1 and idx < len(self.metadata):
                    similarity_score = 1.0 / (1.0 + distance)
                    query_results.append({
                        'rank': i + 1,
                        'distance': float(distance),
                        'score': float(similarity_score),
                        'text': self.chunk_texts[idx],
                        'metadata': self.metadata[idx]
                    })
            all_results.append(query_results)
        return all_results

    def search_with_metadata(self, query_embedding, metadata_filters=None, k=5):
        """
        Hybrid search: first apply metadata filters, then do similarity ranking.
        metadata_filters = {"domain": "technology", "audience": "general"}
        """
        if metadata_filters is None:
            return self.search(query_embedding, k)

        # Filter candidates
        candidate_indices = [
            i for i, m in enumerate(self.metadata)
            if all(m.get(key) == val for key, val in metadata_filters.items())
        ]

        if not candidate_indices:
            return []

        # Reconstruct embeddings for candidates
        candidate_embeddings = np.array([self.index.reconstruct(i) for i in candidate_indices])
        query = query_embedding.astype("float32")

        # Compute L2 distances manually
        distances = np.linalg.norm(candidate_embeddings - query, axis=1)
        sorted_indices = np.argsort(distances)[:k]

        results = []
        for rank, idx in enumerate(sorted_indices, start=1):
            true_idx = candidate_indices[idx]
            distance = float(distances[idx])
            similarity_score = 1.0 / (1.0 + distance)
            results.append({
                "rank": rank,
                "distance": distance,
                "score": similarity_score,
                "text": self.chunk_texts[true_idx],
                "metadata": self.metadata[true_idx]
            })
        return results

    def search_with_intent(self, query_embedding, query_intent, k=5):
        """
        Adjust ranking based on query intent.
        Example query_intent = {"is_definition": True, "is_example": False}
        """
        base_results = self.search(query_embedding, k * 2)

        def intent_score(meta, intent):
            score = 0
            for key, want in intent.items():
                if want and meta.get("intent", {}).get(key):
                    score += 1
            return score

        scored_results = [
            {**res, "intent_score": intent_score(res["metadata"], query_intent)}
            for res in base_results
        ]
        # Re-rank by (intent_score, similarity score)
        scored_results.sort(key=lambda r: (r["intent_score"], r["score"]), reverse=True)
        return scored_results[:k]

    def find_similar_by_id(self, chunk_id, k=5):
        target_idx = None
        for i, metadata in enumerate(self.metadata):
            if metadata.get('chunk_id') == chunk_id:
                target_idx = i
                break
        if target_idx is None:
            return []
        target_embedding = self.index.reconstruct(target_idx)
        target_embedding = np.expand_dims(target_embedding, axis=0)
        return self.search(target_embedding, k)

    def save_index(self, filepath):
        os.makedirs("vector_db", exist_ok=True)
        index_path = os.path.join("vector_db", f"{filepath}.index")
        metadata_path = os.path.join("vector_db", f"{filepath}_metadata.json")

        faiss.write_index(self.index, index_path) #pylint: disable=no-member
        with open(metadata_path, 'w') as f:
            json.dump({
                'metadata': self.metadata,
                'chunk_texts': self.chunk_texts,
                'hnsw_params': {
                    'M': self.M,
                    'ef_construction': self.ef_construction,
                    'ef_search': self.ef_search,
                    'distance_metric': 'euclidean'
                }
            }, f, indent=2)

    def load_index(self, filepath):
        index_path = os.path.join("vector_db", f"{filepath}.index")
        metadata_path = os.path.join("vector_db", f"{filepath}_metadata.json")
        self.index = faiss.read_index(index_path) #pylint: disable=no-member
        self.is_trained = True
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            self.metadata = data['metadata']
            self.chunk_texts = data['chunk_texts']
            if 'hnsw_params' in data:
                self.M = data['hnsw_params'].get('M', self.M)
                self.ef_construction = data['hnsw_params'].get('ef_construction', self.ef_construction)
                self.ef_search = data['hnsw_params'].get('ef_search', self.ef_search)

    def get_index_stats(self):
        if not hasattr(self.index, 'hnsw'):
            return {"error": "Not an HNSW index"}
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "distance_metric": "Euclidean (L2)",
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "max_level": self.index.hnsw.max_level,
            "entry_point": self.index.hnsw.entry_point,
        }
