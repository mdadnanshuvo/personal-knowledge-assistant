# retriever.py
import numpy as np

class Retriever:
    def __init__(self, vector_store):
        self.vs = vector_store

    def retrieve(self, query_data, k=5, max_distance=10.0):
        """
        Args:
            query_data: dict from QueryProcessor.process_query_with_metadata
                        must contain 'embedding', 'metadata_filters', 'query_intent'
        """
        query_embedding = query_data['embedding']
        metadata_filters = query_data.get('metadata_filters', None)
        query_intent = query_data.get('query_intent', None)


        # Step 1: Run base similarity search
        base_results = self.vs.search(query_embedding, k * 3)

        # Step 2: Distance threshold filtering (keep only within max_distance)
        filtered_results = [r for r in base_results if r["distance"] <= max_distance]

        # Step 3: Metadata soft scoring (instead of strict filtering)
        for r in filtered_results:
            if metadata_filters:
                # Count how many metadata keys match
                meta_match_count = sum(
                    1 for key, val in metadata_filters.items()
                    if r["metadata"].get(key) == val
                )
                # Normalize score between 0 and 1
                r["meta_score"] = meta_match_count / len(metadata_filters)
            else:
                r["meta_score"] = 0.0

        # Step 4: Intent re-ranking
        if query_intent:
            def intent_score(meta, intent):
                score = 0
                for key, want in intent.items():
                    if want and meta.get("intent", {}).get(key):
                        score += 1
                return score

            for r in filtered_results:
                r["intent_score"] = intent_score(r["metadata"], query_intent)
        else:
            for r in filtered_results:
                r["intent_score"] = 0

        # Step 5: Weighted scoring
        for r in filtered_results:
            similarity_weight = 0.7
            intent_weight = 0.2
            meta_weight = 0.1

            r["final_score"] = (
                similarity_weight * r["score"]
                + intent_weight * r["intent_score"]
                + meta_weight * r["meta_score"]
            )

        # Sort by final score and return top-k
        return sorted(filtered_results, key=lambda r: r["final_score"], reverse=True)[:k]
