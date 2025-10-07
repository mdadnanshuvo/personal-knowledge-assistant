# chunking/hybrid_chunker.py
import re



class HybridChunker:
     

    def is_qna_format(self,text):
        """
        Check if a given text has Q&A format based on various patterns
        """
        if not text or not isinstance(text, str):
            return False
        
        # Patterns that indicate Q&A format
        patterns = [
            r'Q\.\d+',  # Q.1, Q.2, etc.
            r'Question\s*\d+',  # Question 1, Question 2, etc.
            r'^What\s+',  # Lines starting with "What"
            r'^How\s+',   # Lines starting with "How"  
            r'^Why\s+',   # Lines starting with "Why"
            r'^Where\s+', # Lines starting with "Where"
            r'^When\s+',  # Lines starting with "When"
            r'^Who\s+',   # Lines starting with "Who"
            r'\?$',       # Lines ending with question mark
            r'Answer:',   # Explicit answer indicator
            r'Solution:', # Explicit solution indicator
        ]
        
        # Check for any Q&A patterns in the text
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                return True
        
        return False

    def detect_qna_in_pages(self,pages_data):
        """
        Traverse through pages data and detect if at least 3 out of first 10 pages are in Q&A format
        
        Args:
            pages_data: List of objects with 'text' and 'metadata' keys
        
        Returns:
            bool: True if at least 3 of first 10 pages are Q&A format
        """
        if not pages_data or len(pages_data) < 3:
            return False
        
        # Take first 10 pages or all pages if less than 10
        pages_to_check = pages_data[:10]
        
        qna_count = 0
        qna_pages = []
        
        for page_data in pages_to_check:
            # Extract text from the page object
            text = page_data.get('text', '')
            page_number = page_data.get('metadata', {}).get('page_number', 'unknown')
            
            if self.is_qna_format(text):
                qna_count += 1
                qna_pages.append(page_number)
        
        # Return True if at least 3 pages are Q&A format
        return qna_count >= 3
