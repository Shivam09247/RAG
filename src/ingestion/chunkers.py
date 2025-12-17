"""
Text Chunking Strategies
========================
Implements both semantic and fixed-size chunking strategies.
Semantic chunking > fixed chunks for better retrieval quality.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""
    
    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks."""
        pass
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()


class RecursiveChunker(BaseChunker):
    """
    Recursive character-based text splitter.
    
    Uses a hierarchy of separators to split text while
    maintaining semantic boundaries.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[list[str]] = None,
    ):
        """
        Initialize recursive chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            separators: List of separators in order of preference
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n\n",  # Multiple newlines (section breaks)
            "\n\n",    # Paragraph breaks
            "\n",      # Line breaks
            ". ",      # Sentences
            "! ",      # Exclamations
            "? ",      # Questions
            "; ",      # Semicolons
            ", ",      # Commas
            " ",       # Words
            "",        # Characters
        ]
    
    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )
        
        all_chunks = []
        
        for doc in documents:
            cleaned_text = self._clean_text(doc.page_content)
            chunks = splitter.split_text(cleaned_text)
            
            for i, chunk in enumerate(chunks):
                chunk_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "chunk_count": len(chunks),
                        "chunking_method": "recursive",
                    }
                )
                all_chunks.append(chunk_doc)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks


class SemanticChunker(BaseChunker):
    """
    Semantic text chunker that splits based on meaning.
    
    Uses embeddings to identify semantic boundaries in text,
    resulting in more coherent chunks that preserve context.
    """
    
    def __init__(
        self,
        embeddings,
        breakpoint_threshold: float = 0.5,
        buffer_size: int = 1,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
    ):
        """
        Initialize semantic chunker.
        
        Args:
            embeddings: Embedding model for computing semantic similarity
            breakpoint_threshold: Threshold for semantic discontinuity (0-1)
            buffer_size: Number of sentences to consider for smoothing
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size in characters
        """
        self.embeddings = embeddings
        self.breakpoint_threshold = breakpoint_threshold
        self.buffer_size = buffer_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split documents into semantically coherent chunks."""
        all_chunks = []
        
        for doc in documents:
            cleaned_text = self._clean_text(doc.page_content)
            sentences = self._split_into_sentences(cleaned_text)
            
            if len(sentences) < 3:
                # Too short for semantic chunking
                all_chunks.append(Document(
                    page_content=cleaned_text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": 0,
                        "chunk_count": 1,
                        "chunking_method": "semantic",
                    }
                ))
                continue
            
            # Get embeddings for all sentences
            sentence_embeddings = self.embeddings.embed_documents(sentences)
            
            # Find semantic breakpoints
            breakpoints = self._find_breakpoints(sentence_embeddings)
            
            # Create chunks based on breakpoints
            chunks = self._create_chunks(sentences, breakpoints)
            
            for i, chunk in enumerate(chunks):
                chunk_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "chunk_count": len(chunks),
                        "chunking_method": "semantic",
                    }
                )
                all_chunks.append(chunk_doc)
        
        logger.info(f"Created {len(all_chunks)} semantic chunks from {len(documents)} documents")
        return all_chunks
    
    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _find_breakpoints(self, embeddings: list[list[float]]) -> list[int]:
        """Find semantic breakpoints using cosine similarity."""
        if len(embeddings) < 2:
            return []
        
        embeddings_array = np.array(embeddings)
        
        # Calculate cosine similarities between consecutive sentences
        similarities = []
        for i in range(len(embeddings_array) - 1):
            sim = self._cosine_similarity(embeddings_array[i], embeddings_array[i + 1])
            similarities.append(sim)
        
        # Apply smoothing with buffer
        if self.buffer_size > 0 and len(similarities) > self.buffer_size * 2:
            smoothed = []
            for i in range(len(similarities)):
                start = max(0, i - self.buffer_size)
                end = min(len(similarities), i + self.buffer_size + 1)
                smoothed.append(np.mean(similarities[start:end]))
            similarities = smoothed
        
        # Find breakpoints where similarity drops below threshold
        # Use percentile-based threshold for adaptability
        threshold = np.percentile(similarities, self.breakpoint_threshold * 100)
        
        breakpoints = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                breakpoints.append(i + 1)
        
        return breakpoints
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _create_chunks(self, sentences: list[str], breakpoints: list[int]) -> list[str]:
        """Create chunks from sentences based on breakpoints."""
        if not breakpoints:
            return [" ".join(sentences)]
        
        chunks = []
        start = 0
        
        for bp in breakpoints:
            chunk_text = " ".join(sentences[start:bp])
            
            # Handle chunk size constraints
            if len(chunk_text) >= self.min_chunk_size:
                if len(chunk_text) <= self.max_chunk_size:
                    chunks.append(chunk_text)
                else:
                    # Split oversized chunk
                    sub_chunks = self._split_large_chunk(chunk_text)
                    chunks.extend(sub_chunks)
            else:
                # Merge with next chunk if too small
                if chunks:
                    chunks[-1] += " " + chunk_text
                else:
                    chunks.append(chunk_text)
            
            start = bp
        
        # Handle remaining sentences
        if start < len(sentences):
            remaining = " ".join(sentences[start:])
            if chunks and len(remaining) < self.min_chunk_size:
                chunks[-1] += " " + remaining
            else:
                chunks.append(remaining)
        
        return chunks
    
    def _split_large_chunk(self, text: str) -> list[str]:
        """Split a large chunk that exceeds max size."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1
            if current_length + word_length > self.max_chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks


class MarkdownChunker(BaseChunker):
    """
    Markdown-aware chunker that respects document structure.
    
    Splits on headers while maintaining hierarchy.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split markdown documents by headers."""
        from langchain_text_splitters import MarkdownHeaderTextSplitter
        
        headers_to_split_on = [
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
            ("####", "header_4"),
        ]
        
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )
        
        all_chunks = []
        
        for doc in documents:
            md_chunks = md_splitter.split_text(doc.page_content)
            
            for i, chunk in enumerate(md_chunks):
                chunk_doc = Document(
                    page_content=chunk.page_content,
                    metadata={
                        **doc.metadata,
                        **chunk.metadata,
                        "chunk_index": i,
                        "chunk_count": len(md_chunks),
                        "chunking_method": "markdown",
                    }
                )
                all_chunks.append(chunk_doc)
        
        return all_chunks


def get_chunker(
    chunker_type: str = "semantic",
    embeddings=None,
    **kwargs
) -> BaseChunker:
    """
    Factory function to get the appropriate chunker.
    
    Args:
        chunker_type: Type of chunker ("semantic", "recursive", "markdown")
        embeddings: Embedding model (required for semantic chunking)
        **kwargs: Additional arguments for the chunker
        
    Returns:
        Configured chunker instance
    """
    if chunker_type == "semantic":
        if embeddings is None:
            raise ValueError("Embeddings required for semantic chunking")
        return SemanticChunker(embeddings=embeddings, **kwargs)
    elif chunker_type == "recursive":
        return RecursiveChunker(**kwargs)
    elif chunker_type == "markdown":
        return MarkdownChunker(**kwargs)
    else:
        raise ValueError(f"Unknown chunker type: {chunker_type}")
