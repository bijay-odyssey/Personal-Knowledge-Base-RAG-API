from typing import List
from nltk import sent_tokenize

class Chunker:
    def __init__(self, strategy: str, chunk_size: int = 512):
        self.strategy = strategy
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> List[str]:
        if self.strategy == "fixed":
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        elif self.strategy == "sentence":
            sentences = text.split('. ')  # Basic split
            chunks = []
            current = ""
            for sent in sentences:
                if len(current) + len(sent) > self.chunk_size:
                    chunks.append(current)
                    current = sent + '. '
                else:
                    current += sent + '. '
            if current:
                chunks.append(current)
            return chunks
        elif self.strategy == "recursive":
            paragraphs = text.split('\n\n')
            chunks = []
            for para in paragraphs:
                if len(para) <= self.chunk_size:
                    chunks.append(para)
                else:
                    # Recurse to sentence split
                    chunks.extend(self._sentence_split(para, self.chunk_size))
            return chunks

        raise ValueError("Invalid chunking strategy")

    def _sentence_split(self, text: str, size: int) -> List[str]:
        sentences = sent_tokenize(text)
        chunks = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) > size:
                if current.strip():
                    chunks.append(current.strip())
                current = sent + " "
            else:
                current += sent + " "
        if current.strip():
            chunks.append(current.strip())
        return chunks
