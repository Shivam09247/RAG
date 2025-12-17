"""
Document Loaders
================
Smart document ingestion pipelines supporting multiple file formats.
"""

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Universal document loader supporting multiple file formats.
    
    Supported formats:
    - PDF files
    - Text files (.txt, .md)
    - HTML files
    - Word documents (.docx)
    - Web pages (URL)
    """
    
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".html", ".docx"}
    
    def __init__(self, extract_images: bool = False, extract_tables: bool = True):
        """
        Initialize the document loader.
        
        Args:
            extract_images: Whether to extract images from documents
            extract_tables: Whether to extract tables from documents
        """
        self.extract_images = extract_images
        self.extract_tables = extract_tables
    
    def load(self, source: str | Path) -> list[Document]:
        """
        Load documents from a file path or URL.
        
        Args:
            source: File path or URL to load
            
        Returns:
            List of Document objects
        """
        source_str = str(source)
        
        if source_str.startswith(("http://", "https://")):
            return self._load_url(source_str)
        
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")
        
        if path.is_dir():
            return self._load_directory(path)
        
        return self._load_file(path)
    
    def _load_file(self, path: Path) -> list[Document]:
        """Load a single file."""
        extension = path.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {extension}")
        
        loader_map = {
            ".pdf": self._load_pdf,
            ".txt": self._load_text,
            ".md": self._load_text,
            ".html": self._load_html,
            ".docx": self._load_docx,
        }
        
        loader = loader_map.get(extension)
        if not loader:
            raise ValueError(f"No loader for extension: {extension}")
        
        documents = loader(path)
        
        # Enrich with metadata
        for doc in documents:
            doc.metadata.update(self._get_file_metadata(path))
        
        logger.info(f"Loaded {len(documents)} documents from {path}")
        return documents
    
    def _load_directory(self, directory: Path) -> list[Document]:
        """Load all supported files from a directory."""
        documents = []
        
        for ext in self.SUPPORTED_EXTENSIONS:
            for file_path in directory.rglob(f"*{ext}"):
                try:
                    docs = self._load_file(file_path)
                    documents.extend(docs)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
        
        return documents
    
    def _load_pdf(self, path: Path) -> list[Document]:
        """Load PDF file using PyPDF."""
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(str(path))
            return loader.load()
        except ImportError:
            # Fallback to basic PDF loading
            from pypdf import PdfReader
            
            reader = PdfReader(str(path))
            documents = []
            
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={"page": i + 1, "source": str(path)}
                    ))
            
            return documents
    
    def _load_text(self, path: Path) -> list[Document]:
        """Load text file."""
        content = path.read_text(encoding="utf-8")
        return [Document(page_content=content, metadata={"source": str(path)})]
    
    def _load_html(self, path: Path) -> list[Document]:
        """Load HTML file."""
        try:
            from langchain_community.document_loaders import BSHTMLLoader
            loader = BSHTMLLoader(str(path))
            return loader.load()
        except ImportError:
            from bs4 import BeautifulSoup
            
            content = path.read_text(encoding="utf-8")
            soup = BeautifulSoup(content, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            
            return [Document(page_content=text, metadata={"source": str(path)})]
    
    def _load_docx(self, path: Path) -> list[Document]:
        """Load Word document."""
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(str(path))
            return loader.load()
        except ImportError:
            raise ImportError("docx2txt is required for .docx files")
    
    def _load_url(self, url: str) -> list[Document]:
        """Load content from URL."""
        try:
            from langchain_community.document_loaders import WebBaseLoader
            loader = WebBaseLoader(url)
            return loader.load()
        except ImportError:
            import httpx
            from bs4 import BeautifulSoup
            
            response = httpx.get(url, follow_redirects=True, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text(separator="\n", strip=True)
            
            return [Document(page_content=text, metadata={"source": url})]
    
    def _get_file_metadata(self, path: Path) -> dict[str, Any]:
        """Extract metadata from file."""
        stat = path.stat()
        
        # Calculate file hash for deduplication
        file_hash = hashlib.md5(path.read_bytes()).hexdigest()
        
        return {
            "file_name": path.name,
            "file_path": str(path.absolute()),
            "file_size": stat.st_size,
            "file_type": path.suffix.lower(),
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "file_hash": file_hash,
            "ingested_at": datetime.now().isoformat(),
        }


class WebCrawler:
    """Crawl and load content from websites."""
    
    def __init__(self, max_depth: int = 2, max_pages: int = 50):
        """
        Initialize web crawler.
        
        Args:
            max_depth: Maximum crawl depth
            max_pages: Maximum number of pages to crawl
        """
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited: set[str] = set()
    
    async def crawl(self, start_url: str) -> list[Document]:
        """
        Crawl website starting from URL.
        
        Args:
            start_url: Starting URL for crawl
            
        Returns:
            List of Document objects
        """
        import httpx
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin, urlparse
        
        documents = []
        to_visit = [(start_url, 0)]
        
        async with httpx.AsyncClient(timeout=30) as client:
            while to_visit and len(documents) < self.max_pages:
                url, depth = to_visit.pop(0)
                
                if url in self.visited or depth > self.max_depth:
                    continue
                
                self.visited.add(url)
                
                try:
                    response = await client.get(url, follow_redirects=True)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, "html.parser")
                    
                    # Extract text
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    text = soup.get_text(separator="\n", strip=True)
                    
                    if text:
                        documents.append(Document(
                            page_content=text,
                            metadata={
                                "source": url,
                                "title": soup.title.string if soup.title else "",
                                "depth": depth,
                            }
                        ))
                    
                    # Find links for further crawling
                    if depth < self.max_depth:
                        base_domain = urlparse(start_url).netloc
                        
                        for link in soup.find_all("a", href=True):
                            href = urljoin(url, link["href"])
                            parsed = urlparse(href)
                            
                            # Only follow links on same domain
                            if parsed.netloc == base_domain:
                                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                                if clean_url not in self.visited:
                                    to_visit.append((clean_url, depth + 1))
                
                except Exception as e:
                    logger.warning(f"Failed to crawl {url}: {e}")
        
        return documents
