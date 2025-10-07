class SlidingWindowChunker:
    def __init__(self, chunk_size=600, overlap=200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str):
        """
        Break text into overlapping text
        """
        if not text:
            return

        tokens = text.split()
        chunks = []
        start = 0

        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunks.append(" ".join(chunk_tokens))
            start += self.chunk_size - self.overlap
        return chunks
