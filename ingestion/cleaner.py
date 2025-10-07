import re
from collections import Counter


class TextCleaner:
    def __init__(self):
        # Enhanced regex patterns for cleaning noisy characters
        self.patterns = {
            # (cid:27), (cid:7) etc.
            "cid_pattern": re.compile(r"\(cid:\d+\)"),
            # non-printable chars
            "control_chars": re.compile(r"[\x00-\x1f\x7f-\x9f]"),
            # hyphenated line breaks
            "hyphen_breaks": re.compile(r"(\w+)-\s*\n\s*(\w+)"),
            # multiple spaces/newlines
            "multi_whitespace": re.compile(r"\s+"),
            # decorative lines
            "boilerplate": re.compile(r"={3,}|-{3,}|\*{3,}"),
            # single orphan characters
            "orphan_chars": re.compile(r"\s+[A-Za-z]\s+"),
            # website URLs
            "url_pattern": re.compile(r"www\.\S+\.com"),
            # standalone page numbers
            "page_numbers": re.compile(r"^\s*\d+\s*$", re.MULTILINE),
            # bullet point artifacts
            "bullet_noise": re.compile(r"[•·∙●○▪▫►◄]", re.MULTILINE),
            # non-ASCII characters that might be noise
            "unicode_noise": re.compile(r"[^\x00-\x7F]+"),
            # repeated punctuation
            "repeated_punctuation": re.compile(r"([!?.,])\1+"),
        }

        # Common OCR artifacts to remove
        self.ocr_artifacts = {
            # FA, ZA, tA patterns
            "fa_za_pattern": re.compile(r"\b[FZt]A\s+"),
            # specific symbols from your example
            "weird_symbols": re.compile(r"[¬Ä‘…²Ùö]", re.MULTILINE),
            # other common symbol artifacts
            "cid_like_symbols": re.compile(r"[©®™§¶]", re.MULTILINE),
        }

    def clean_with_metadata(self, text: str, page_delimiter: str = "\f") -> dict:
        """
        Enhanced text cleaning with better OCR artifact removal.
        """
        if not text or not isinstance(text, str):
            return {"pages": [], "metadata": {}}

        # Split into pages
        pages = text.split(
            page_delimiter) if page_delimiter in text else [text]

        first_lines, last_lines = [], []
        cleaned_pages = []

        # First pass: collect first/last lines for repeated header/footer detection
        for page in pages:
            lines = page.strip().splitlines()
            if not lines:
                continue
            first_lines.append(lines[0].strip())
            last_lines.append(lines[-1].strip())

        # Detect repeated headers/footers globally
        global_header = self._find_repeated(first_lines)
        global_footer = self._find_repeated(last_lines)

        # Process each page individually with enhanced cleaning
        for page_num, page in enumerate(pages, start=1):
            lines = page.strip().splitlines()
            if not lines:
                continue

            # Enhanced cleaning for each line before joining
            cleaned_lines = []
            for line in lines:
                cleaned_line = self._apply_enhanced_cleaning(line)
                if cleaned_line.strip():  # Only keep non-empty lines after cleaning
                    cleaned_lines.append(cleaned_line)

            cleaned_text = "\n".join(cleaned_lines)

            # Page-level metadata - only include non-null values
            page_metadata = {
                "page_number": page_num,
                "char_count": len(cleaned_text),
                "line_count": len(cleaned_lines),
            }

            # Only add header if it exists and matches global header
            if (page_num-1 < len(first_lines) and 
                first_lines[page_num-1] == global_header and 
                global_header is not None):
                page_metadata["header"] = first_lines[page_num-1]

            # Only add footer if it exists and matches global footer
            if (page_num-1 < len(last_lines) and 
                last_lines[page_num-1] == global_footer and 
                global_footer is not None):
                page_metadata["footer"] = last_lines[page_num-1]

            # Enhanced section title detection - only add if found
            section_title = self._detect_section_title(
                cleaned_lines, global_header, global_footer)
            if section_title:
                page_metadata["section_title"] = section_title

            cleaned_pages.append({
                "text": cleaned_text,
                "metadata": page_metadata
            })

        # Global metadata extraction - only non-null values
        metadata = self._extract_global_metadata(
            pages, global_header, global_footer)

        return {"pages": cleaned_pages, "metadata": metadata}

    def _apply_enhanced_cleaning(self, text: str) -> str:
        """Apply comprehensive cleaning rules to remove OCR noise and artifacts"""

        # Remove CID patterns first (most common OCR artifact)
        text = self.patterns["cid_pattern"].sub(" ", text)

        # Remove specific OCR artifacts from your example
        text = self.ocr_artifacts["fa_za_pattern"].sub(" ", text)
        text = self.ocr_artifacts["weird_symbols"].sub(" ", text)
        text = self.ocr_artifacts["cid_like_symbols"].sub(" ", text)

        # Standard cleaning pipeline
        text = self.patterns["control_chars"].sub(" ", text)
        text = self.patterns["unicode_noise"].sub(" ", text)
        text = self.patterns["bullet_noise"].sub(" ", text)
        text = self.patterns["hyphen_breaks"].sub(r"\1\2", text)
        text = self.patterns["boilerplate"].sub(" ", text)
        text = self.patterns["url_pattern"].sub(" ", text)
        text = self.patterns["page_numbers"].sub(" ", text)
        text = self.patterns["orphan_chars"].sub(" ", text)
        text = self.patterns["repeated_punctuation"].sub(r"\1", text)
        text = self.patterns["multi_whitespace"].sub(" ", text)

        # Final cleanup: remove extra spaces and trim
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _detect_section_title(self, lines: list, global_header: str, global_footer: str) -> str:
        """Enhanced section title detection"""
        for line in lines[:5]:  # Check first 5 lines
            clean_line = line.strip()
            if not clean_line:
                continue

            # Skip if it's a header/footer
            if clean_line in [global_header, global_footer]:
                continue

            # Section title patterns (improved heuristics)
            is_section = (
                # All caps with reasonable length (not too short, not too long)
                (clean_line.isupper() and 2 <= len(clean_line) <= 50) or
                # Starts with common section prefixes
                re.match(r"^(CHAPTER|SECTION|PART|TOPIC)\s+\d+", clean_line, re.I) or
                # Difficulty levels (EASY, MEDIUM, HARD)
                re.match(r"^(EASY|MEDIUM|HARD|BEGINNER|INTERMEDIATE|ADVANCED)\b", clean_line, re.I) or
                # Question patterns (Q.1, Q1, Question 1)
                re.match(r"^Q\.?\d+", clean_line)
            )

            if is_section:
                return clean_line

        return None

    def _extract_global_metadata(self, pages: list, global_header: str, global_footer: str) -> dict:
        """Enhanced global metadata extraction - only returns non-null values"""
        metadata = {}

        # Only add header/footer if they exist
        if global_header:
            metadata["header"] = global_header
        if global_footer:
            metadata["footer"] = global_footer

        if not pages:
            return metadata

        first_page_lines = pages[0].splitlines()

        # Enhanced title detection - only add if found
        for line in first_page_lines[:5]:
            clean_line = line.strip()
            if (clean_line and
                clean_line not in [global_header, global_footer] and
                "by" not in clean_line.lower() and
                len(clean_line) > 5 and  # Reasonable title length
                    not clean_line.isupper()):  # Titles usually not ALL CAPS
                metadata["title"] = clean_line
                break

        # Enhanced author detection - only add if found
        for line in first_page_lines[:10]:
            author_match = re.search(
                r"by\s+([A-Za-z\.\s]+)(?:\s+and\s+([A-Za-z\.\s]+))?", line, re.I)
            if author_match:
                authors = [author_match.group(1).strip()]
                if author_match.group(2):
                    authors.append(author_match.group(2).strip())
                metadata["author"] = " and ".join(authors)
                break

        # Enhanced year detection - only add if found
        for line in first_page_lines[:10]:
            year_match = re.search(r"\b(19|20)\d{2}\b", line)
            if year_match:
                metadata["year"] = year_match.group()
                break

        return metadata

    def _find_repeated(self, lines, min_repeats=3):
        """Detect lines repeated across pages (used for header/footer)"""
        counter = Counter([l for l in lines if l and len(
            l.strip()) > 2])  # Only consider substantial lines
        if not counter:
            return None
        most_common, count = counter.most_common(1)[0]
        return most_common if count >= min_repeats else None

    def batch_clean(self, texts: list, page_delimiter: str = "\f") -> list:
        """Clean multiple texts in batch"""
        return [self.clean_with_metadata(text, page_delimiter) for text in texts]


# Example usage:
if __name__ == "__main__":
    cleaner = TextCleaner()
    
    # Example text with CID patterns
    sample_text = """JAVASCRIPT Concepts to Ace Technical Interview\f
EASY How do you detect primitive or non- Q.1 primitive value types in Javascript? (cid:7) This operator returns... (cid:27)\f
EASY Explain the key features introduced in Q.2 Javascript ES6"""
    
    result = cleaner.clean_with_metadata(sample_text)
    
    print("Cleaned pages:")
    for page in result["pages"]:
        print(f"Page {page['metadata']['page_number']}:")
        print(f"  Text: {page['text'][:100]}...")
        print(f"  Metadata: {page['metadata']}")
        print()
    
    print("Global metadata:")
    print(result["metadata"])