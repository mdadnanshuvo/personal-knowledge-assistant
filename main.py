import time
from rag.generation import Generator
from rag.query_processor import QueryProcessor
from utils.document_processor import process_all_documents
from utils.embedding_utils import embed_chunks_with_metadata
from utils.faiss_utils import FAISSUtils
from rag.retriever import Retriever

if __name__ == "__main__":
    start_time = time.time()
    
    # Process documents
    print("Processing documents...")
    all_chunks_with_metadata = process_all_documents("docs")

    # Embed chunks
    embedded_chunks = embed_chunks_with_metadata(all_chunks_with_metadata)

    # Query processing
    pq = QueryProcessor()
    query = "What was the deadline of submitting Scrappy assignment?"
    query_data = pq.process_query_with_metadata(query) 
    
    
    # Build FAISS index
    faiss_utils = FAISSUtils()
    faiss_store = faiss_utils.build_faiss_index(embedded_chunks, "my_knowledge_base")

    # Initialize Retriever
    r = Retriever(faiss_store)

    # Retrieve with metadata + intent
    results = r.retrieve(query_data, k=5)

    g = Generator()

    print(g.generate(query, results))
