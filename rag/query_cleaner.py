# rag/query_cleaner.py
import re
import string

from ingestion.cleaner import TextCleaner

class QueryCleaner:
    """
    Specialized cleaner for user queries that matches document cleaning pipeline
    """
    
    def __init__(self):
        self.cleaner = TextCleaner()  # Your existing cleaner
    
    def clean(self, query: str) -> str:
        """
        Clean user query with the same pipeline as documents
        """
        # Basic cleaning
        cleaned = query.strip()
        
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Remove excessive punctuation (keep basic punctuation for question queries)
        cleaned = re.sub(r'[^\w\s?]', '', cleaned)
        
        # Normalize case (optional - depends on your embedding model)
        # Most modern embedding models are case-insensitive
        cleaned = cleaned.lower()
        
        # Remove very short queries or handle edge cases
        if len(cleaned) < 2:
            return cleaned
            
        # Ensure the query ends properly (for question queries)
        if cleaned and not cleaned[-1] in ['.', '?', '!']:
            cleaned = cleaned  # Keep as is, or add period if needed
        
        return cleaned
    
    def clean_for_embedding(self, query: str) -> str:
        """
        Special cleaning optimized for embedding generation
        """
        cleaned = self.clean(query)
        
        # Additional steps specific for embedding quality
        # Remove any remaining HTML tags if any
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        
        # Normalize whitespace characters
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()