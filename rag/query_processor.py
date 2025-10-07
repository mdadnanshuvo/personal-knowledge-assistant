from typing import Dict
import numpy as np
from embedding.embedder import Embedders
from rag.query_cleaner import QueryCleaner
import re

class QueryProcessor:
    def __init__(self):
        self.embedder = Embedders()
        self.cleaner = QueryCleaner()
    
    def extract_metadata_filters(self, query: str) -> Dict:
        """
        Extract universal metadata filters from user query
        """
        filters = {}
        query_lower = query.lower()

        # ---------------------------
        # Document Type
        # ---------------------------
        doc_types = {
            'official': ['official', 'documentation', 'docs', 'manual', 'spec'],
            'tutorial': ['tutorial', 'guide', 'walkthrough', 'how-to', 'lesson'],
            'example': ['example', 'sample', 'snippet', 'template'],
            'reference': ['reference', 'cheatsheet', 'api reference', 'syntax'],
            'faq': ['faq', 'frequently asked'],
            'research': ['research', 'paper', 'study', 'analysis'],
            'news': ['news', 'article', 'blog', 'update', 'release notes'],
            'report': ['report', 'case study', 'whitepaper'],
        }
        for doc_type, keywords in doc_types.items():
            if any(keyword in query_lower for keyword in keywords):
                filters['doc_type'] = doc_type
                break

        # ---------------------------
        # Recency
        # ---------------------------
        if any(word in query_lower for word in ['latest', 'recent', 'current', 'new']):
            filters['time'] = 'recent'
        elif any(word in query_lower for word in ['old', 'archived', 'legacy', 'historical']):
            filters['time'] = 'historical'

        # ---------------------------
        # Complexity / Depth
        # ---------------------------
        if any(word in query_lower for word in ['basic', 'beginner', 'introduction', 'getting started']):
            filters['complexity'] = 'beginner'
        elif any(word in query_lower for word in ['intermediate', 'moderate']):
            filters['complexity'] = 'intermediate'
        elif any(word in query_lower for word in ['advanced', 'expert', 'comprehensive', 'deep']):
            filters['complexity'] = 'advanced'

        # ---------------------------
        # Task / Intent
        # ---------------------------
        if any(word in query_lower for word in ['what is', 'define', 'definition']):
            filters['intent'] = 'definition'
        elif any(word in query_lower for word in ['how to', 'steps', 'procedure', 'tutorial']):
            filters['intent'] = 'how-to'
        elif any(word in query_lower for word in ['vs', 'versus', 'compare', 'difference']):
            filters['intent'] = 'comparison'
        elif any(word in query_lower for word in ['example', 'sample', 'code']):
            filters['intent'] = 'example'
        elif any(word in query_lower for word in ['error', 'fix', 'troubleshoot', 'problem', 'issue']):
            filters['intent'] = 'troubleshooting'
        elif any(word in query_lower for word in ['install', 'setup', 'configure']):
            filters['intent'] = 'installation'
        elif any(word in query_lower for word in ['deploy', 'production', 'hosting']):
            filters['intent'] = 'deployment'
        elif any(word in query_lower for word in ['optimize', 'performance', 'scaling']):
            filters['intent'] = 'optimization'
        elif any(word in query_lower for word in ['security', 'vulnerability', 'auth']):
            filters['intent'] = 'security'

        # ---------------------------
        # Domain / Topic
        # ---------------------------
        domains = {
            'technology': ['software', 'ai', 'ml', 'cloud', 'api', 'database', 'programming', 'coding'],
            'science': ['biology', 'physics', 'chemistry', 'neuroscience'],
            'health': ['medical', 'doctor', 'medicine', 'disease', 'treatment'],
            'finance': ['banking', 'investment', 'trading', 'economy'],
            'legal': ['law', 'regulation', 'compliance'],
            'education': ['curriculum', 'learning', 'school', 'university'],
            'business': ['marketing', 'sales', 'startup', 'management'],
            'general': ['general', 'overview']
        }
        for domain, keywords in domains.items():
            if any(keyword in query_lower for keyword in keywords):
                filters['domain'] = domain
                break

        # ---------------------------
        # Audience
        # ---------------------------
        if any(word in query_lower for word in ['student', 'beginner', 'learner']):
            filters['audience'] = 'students'
        elif any(word in query_lower for word in ['professional', 'engineer', 'developer', 'manager']):
            filters['audience'] = 'professionals'
        elif any(word in query_lower for word in ['research', 'scholar', 'scientist']):
            filters['audience'] = 'researchers'
        else:
            filters['audience'] = 'general'

        # ---------------------------
        # Format
        # ---------------------------
        if any(word in query_lower for word in ['pdf']):
            filters['format'] = 'pdf'
        elif any(word in query_lower for word in ['ppt', 'presentation']):
            filters['format'] = 'ppt'
        elif any(word in query_lower for word in ['doc', 'word']):
            filters['format'] = 'doc'
        elif any(word in query_lower for word in ['markdown', 'md']):
            filters['format'] = 'markdown'
        elif any(word in query_lower for word in ['html', 'webpage']):
            filters['format'] = 'html'
        elif any(word in query_lower for word in ['blog', 'article']):
            filters['format'] = 'blog'

        return filters

    
    def clean_query(self, query: str) -> str:
        """
        Clean query while preserving intent
        """
        return self.cleaner.clean_for_embedding(query)
    
    def process_query_with_metadata(self, query: str) -> Dict:
        """
        Process query and extract metadata preferences
        """
        cleaned_query = self.clean_query(query)
        metadata_filters = self.extract_metadata_filters(query)
        
        print(f"Original query: '{query}'")
        print(f"Cleaned query: '{cleaned_query}'")
        print(f"Metadata filters: {metadata_filters}")
        
        # Embed the cleaned query
        query_embedding = self.embedder.Embedding([cleaned_query])[0].astype('float32')
        query_embedding = np.expand_dims(query_embedding, axis=0)
        
        return {
            'original_query': query,
            'cleaned_query': cleaned_query,
            'embedding': query_embedding,
            'metadata_filters': metadata_filters,
            'query_intent': self.analyze_query_intent(query)
        }
    
  

    def analyze_query_intent(self, query: str) -> Dict:
        """
        Analyze query for deeper intent understanding using keywords + regex
        """
        query_lower = query.lower()
        
        # Regex patterns for QnA / definitions / questions
        question_patterns = [
            r'Q\.\d+',             # Matches Q.1, Q.2 ...
            r'Question\s*\d+',     # Matches Question 1, Question 2
            r'^\s*what\s+',        # Line starts with What
            r'^\s*how\s+',         # Line starts with How
            r'^\s*why\s+',         # Line starts with Why
            r'^\s*where\s+',       # Line starts with Where
            r'^\s*when\s+',        # Line starts with When
            r'^\s*who\s+',         # Line starts with Who
            r'\?$',                # Ends with ?
        ]
        
        answer_patterns = [
            r'Answer:',            # Explicit Answer
            r'Solution:',          # Explicit Solution
            r'Explanation:',       # Explicit Explanation
        ]
        
        def matches_any(patterns, text):
            return any(re.search(p, text, re.IGNORECASE) for p in patterns)
        
        return {
            # Core intents (keyword-based)
            'is_definition': any(word in query_lower for word in [
                'what is', 'define', 'definition', 'meaning of', 'explain', 'describe'
            ]) or matches_any(question_patterns, query),
            
            'is_howto': any(word in query_lower for word in [
                'how to', 'how do i', 'steps to', 'procedure', 'guide to'
            ]) or re.match(r'^\s*how\s+', query, re.IGNORECASE),
            
            'is_comparison': any(word in query_lower for word in [
                'vs', 'versus', 'compare', 'difference', 'advantages', 'disadvantages'
            ]),
            
            'is_example': any(word in query_lower for word in [
                'example', 'sample', 'code snippet', 'template'
            ]),
            
            'is_troubleshooting': any(word in query_lower for word in [
                'error', 'fix', 'issue', 'problem', 'bug', 'troubleshoot', 'solution'
            ]) or matches_any(answer_patterns, query),
            
            'is_explanation': any(word in query_lower for word in [
                'explain', 'overview', 'summary', 'details'
            ]) or 'explanation:' in query_lower,
            
            'requires_depth': any(word in query_lower for word in [
                'detailed', 'comprehensive', 'in-depth', 'thorough'
            ]),
            
            'requires_basics': any(word in query_lower for word in [
                'basic', 'beginner', 'simple', 'introduction', 'for dummies'
            ]),
        }
