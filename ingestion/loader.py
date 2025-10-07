
from pathlib import Path
import numpy as np
import pdfplumber
import docx
from bs4 import BeautifulSoup
import json
import fitz

# Optional OCR support
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image, ImageEnhance
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning! OCR depedencies not installed. Scanned PDf will not work.")


class DocumentLoader:
    def __init__(self, docs_path="docs"):
        self.docs_path = Path(docs_path)
        if not self.docs_path.exists():
            self.docs_path.mkdir(parents=True, exist_ok=True)

    def load_documents(self):
        """
        Load all the documents from the docs folder.
        Returns a list of tuples : (filenames, text)
        """

        documents = []

        for file_path in self.docs_path.iterdir():
            if file_path.is_file():
                ext = file_path.suffix.lower()
                text = ""

                try:
                    if ext == ".pdf":
                        text = self._load_pdf(file_path)
                    elif ext in [".docx", ".doc"]:
                        text = self._load_docx(file_path)
                    elif ext in [".md", ".txt"]:
                        text = self._load_text(file_path)
                    elif ext == ".json":
                        text = self._load_json(file_path)
                    else:
                        print(f"Unsupported file format: {file_path}")
                        continue
                except Exception as e:
                    print(f"Error loading {file_path.name}: {e}")
                    continue
                documents.append((file_path.name, text))

        return documents

    def _load_pdf(self, file_path):
        text = ""

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    # 1. Try pdfplumber first
                    normal_text = page.extract_text() or ""

                    # 2. If empty, try PyMuPDF
                    if not normal_text.strip():
                        try:
                            doc = fitz.open(file_path)
                            normal_text = doc[page_num-1].get_text("text") or ""
                        except Exception as e:
                            print(f"PyMuPDF failed on {file_path.name}, page {page_num}: {e}")

                    # 3. If still empty, fallback to OCR
                    ocr_text = ""
                    if not normal_text.strip() and OCR_AVAILABLE:
                        pil_image = convert_from_path(
                            file_path, dpi=300,
                            first_page=page_num, last_page=page_num
                        )[0]
                        ocr_text = self._ocr_image(pil_image)

                    combined_text = (normal_text.strip() + "\n" + ocr_text.strip()).strip()
                    text += combined_text + "\f"

        except Exception as e:
            print(f"Error reading PDF {file_path.name}: {e}")

        return text
    
    def _ocr_image(self, img):
        img = img.convert("L")
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2)
        # OCR
        return pytesseract.image_to_string(img, lang="eng")

    def _load_docx(self, file_path):
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
 
    def _load_text(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_html(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            return soup.get_text(separator="\n")

    def _load_json(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return self._extract_from_json(data)

    def _extract_from_json(self, obj):
        if isinstance(obj, dict):
            parts = []
            for k, v in obj.items():
                parts.append(f"{k} : {self._extract_from_json(v)}")
            return " ".join(parts)
        elif isinstance(obj, list):
            return " ".join([self._extract_from_json(i) for i in obj])
        else:
            return str(obj)
