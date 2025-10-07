import re
from typing import List, Callable

class RecursiveChunker:
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 100,
                 length_function: Callable[[str], int] = len,
                 separators: List[str] = None):
        """
        Improved recursive chunking with standard features.
        
        Args:
            chunk_size: Maximum size of chunks (in characters)
            chunk_overlap: Overlap between chunks (in characters) 
            min_chunk_size: Minimum chunk size to keep
            length_function: Function to calculate text length
            separators: List of separators to try in order
        """
        if chunk_overlap > chunk_size:
            raise ValueError("chunk_overlap must be <= chunk_size")
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.length_function = length_function
        self.separators = separators or [
            "\n\n",          # Double newlines (paragraphs)
            "\n",            # Single newlines  
            r"\.\s+",        # Sentences (period + whitespace)
            r"\?\s+",        # Questions  
            r"!\s+",         # Exclamations
            r";\s+",         # Semicolons
            r",\s+",         # Commas
            r"\s+",          # Whitespace (words)
            ""               # Characters (fallback)
        ]

    def chunk(self, text: str) -> List[str]:
        """Split text into chunks recursively."""
        if not text or self.length_function(text) <= self.chunk_size:
            return [text] if text else []
            
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators."""
        if not separators:
            return self._split_text(text)
            
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split by current separator
        splits = self._split_by_separator(text, separator)
        
        chunks = []
        current_chunk = ""
        
        for split in splits:
            split_len = self.length_function(split)
            
            # If adding this split would exceed chunk size
            if (self.length_function(current_chunk) + split_len > self.chunk_size and 
                self.length_function(current_chunk) >= self.min_chunk_size):
                
                # Add current chunk to results
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap from previous
                if self.chunk_overlap > 0:
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + " " + split
                else:
                    current_chunk = split
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += " " + split if separator != "" else split
                else:
                    current_chunk = split
        
        # Add final chunk
        if current_chunk and self.length_function(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())
            
        # Recursively split chunks that are still too large
        final_chunks = []
        for chunk in chunks:
            if (self.length_function(chunk) > self.chunk_size and 
                len(remaining_separators) > 0):
                final_chunks.extend(self._recursive_split(chunk, remaining_separators))
            else:
                final_chunks.append(chunk)
                
        return final_chunks

    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """Split text by separator while preserving the separator in results."""
        if separator == "":
            return list(text)  # Character-level splitting
            
        # Use regex to split while keeping separators
        if separator in ["\n\n", "\n"]:
            # For newlines, split and keep the newline with content
            parts = re.split(f'({separator})', text)
            # Recombine separators with following content
            result = []
            i = 0
            while i < len(parts):
                if i + 1 < len(parts) and parts[i+1] == separator:
                    result.append(parts[i] + parts[i+1])
                    i += 2
                else:
                    if parts[i]:  # Skip empty strings
                        result.append(parts[i])
                    i += 1
            return result
        else:
            # For other separators, use standard split
            return [part.strip() for part in re.split(separator, text) if part.strip()]

    def _get_overlap_text(self, text: str) -> str:
        """Get overlapping text from the end of a chunk."""
        words = text.split()
        overlap_words = max(1, int(len(words) * (self.chunk_overlap / self.chunk_size)))
        return " ".join(words[-overlap_words:])

    def _split_text(self, text: str) -> List[str]:
        """Final fallback splitting when no separators work."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            if end < len(text):
                # Try to break at word boundary
                while end > start and text[end] not in ' \t\n':
                    end -= 1
                if end == start:  # No word boundary found
                    end = start + self.chunk_size
            else:
                end = len(text)
                
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                
            start = end - self.chunk_overlap if self.chunk_overlap > 0 else end
            
        return chunks