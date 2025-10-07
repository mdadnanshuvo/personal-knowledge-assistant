# utils/document_processor.py
import json
from typing import List, Dict, Tuple
from ingestion.loader import DocumentLoader
from ingestion.cleaner import TextCleaner
from chunking.chunking_pipeline import HybridChunker
from chunking.sliding_window import SlidingWindowChunker
from chunking.recursive import RecursiveChunker

def process_all_documents(docs_path: str) -> List[Dict]:
    """
    Process all documents and return chunks with metadata
    """
    dl = DocumentLoader(docs_path)
    cleaner = TextCleaner()
    chunker = HybridChunker()
    sliding_chunker = SlidingWindowChunker()
    recursive_chunker = RecursiveChunker()
    
    data = dl.load_documents()
    all_chunks_with_metadata = []
    
    for filename, raw_text in data:
        cleaned = cleaner.clean_with_metadata(raw_text)
        chunk_count = 1
        
        use_qna = chunker.detect_qna_in_pages(cleaned.get("pages", []))

        for page in cleaned.get("pages", []):
            page_text = page.get("text", "")
            if not page_text:
                continue

            # Use the selected strategy
            if use_qna:
                chunks = sliding_chunker.chunk(page_text)
            else:
                chunks = recursive_chunker.chunk(page_text)
            
            # Store chunks with metadata
            for chunk_text in chunks:
                chunk_data = {
                    'text': chunk_text,
                    'metadata': {
                        'chunk_id': chunk_count,
                        'page_number': page['metadata'].get('page_number', 'N/A'),
                        'filename': filename,
                        'source_document': filename,
                        'chunking_strategy': 'qna' if use_qna else 'recursive',
                        **page['metadata']
                    }
                }
                all_chunks_with_metadata.append(chunk_data)
                chunk_count += 1
    
    return all_chunks_with_metadata

