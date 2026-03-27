"""Document chunking utilities."""

import re
from typing import List, Dict, Any
import PyPDF2


class DocumentChunker:
    """Splits documents into atomic chunks for retrieval."""
    
    def __init__(self, min_words: int = 50, max_words: int = 300):
        """Initialize chunker with word limits."""
        self.min_words = min_words
        self.max_words = max_words
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            return ""
    
    def chunk_markdown(self, content: str, source: str, topic: str) -> List[Dict[str, Any]]:
        """Chunk markdown content into atomic pieces."""
        chunks = []
        
        # Split by headings
        sections = re.split(r'\n##\s+|\n#\s+', content)
        
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            
            words = section.split()
            
            if len(words) < self.min_words:
                continue
            
            if len(words) > self.max_words:
                paragraphs = re.split(r'\n\n+', section)
                for j, para in enumerate(paragraphs):
                    para_words = para.split()
                    if self.min_words <= len(para_words) <= self.max_words:
                        chunks.append({
                            "chunk_text": para.strip(),
                            "metadata": {
                                "topic": topic,
                                "source": source,
                                "chunk_id": f"{source}_{i}_{j}",
                                "difficulty": "intermediate",
                                "type": "concept_explanation"
                            }
                        })
            else:
                chunks.append({
                    "chunk_text": section.strip(),
                    "metadata": {
                        "topic": topic,
                        "source": source,
                        "chunk_id": f"{source}_{i}",
                        "difficulty": "intermediate",
                        "type": "concept_explanation"
                    }
                })
        
        return chunks
    
    def chunk_text(self, content: str, source: str, topic: str) -> List[Dict[str, Any]]:
        """Chunk plain text content."""
        chunks = []
        
        # Split by paragraphs
        paragraphs = re.split(r'\n\n+', content)
        
        for i, para in enumerate(paragraphs):
            if not para.strip():
                continue
            
            words = para.split()
            
            if len(words) < self.min_words:
                continue
            
            if len(words) > self.max_words:
                # Split long paragraphs into sentences
                sentences = re.split(r'[.!?]+', para)
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk.split()) + len(sentence.split()) <= self.max_words:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk.strip():
                            chunks.append({
                                "chunk_text": current_chunk.strip(),
                                "metadata": {
                                    "topic": topic,
                                    "source": source,
                                    "chunk_id": f"{source}_{i}",
                                    "difficulty": "intermediate",
                                    "type": "concept_explanation"
                                }
                            })
                            current_chunk = sentence + ". "
                if current_chunk.strip():
                    chunks.append({
                        "chunk_text": current_chunk.strip(),
                        "metadata": {
                            "topic": topic,
                            "source": source,
                            "chunk_id": f"{source}_{i}",
                            "difficulty": "intermediate",
                            "type": "concept_explanation"
                        }
                    })
            else:
                chunks.append({
                    "chunk_text": para.strip(),
                    "metadata": {
                        "topic": topic,
                        "source": source,
                        "chunk_id": f"{source}_{i}",
                        "difficulty": "intermediate",
                        "type": "concept_explanation"
                    }
                })
        
        return chunks
    
    def chunk_pdf(self, pdf_file, source: str, topic: str) -> List[Dict[str, Any]]:
        """Extract and chunk PDF content."""
        content = self.extract_text_from_pdf(pdf_file)
        if content:
            return self.chunk_text(content, source, topic)
        return []
    
    def process_file(self, file, source: str, topic: str) -> List[Dict[str, Any]]:
        """Process different file types."""
        file_extension = source.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            return self.chunk_pdf(file, source, topic)
        elif file_extension == 'md':
            content = file.read().decode('utf-8')
            return self.chunk_markdown(content, source, topic)
        elif file_extension == 'txt':
            content = file.read().decode('utf-8')
            return self.chunk_text(content, source, topic)
        else:
            return []
