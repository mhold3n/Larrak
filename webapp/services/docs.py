"""
Documentation service for the Larrak Dashboard.
Provides file tree scanning and markdown-to-HTML rendering.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import markdown
from markdown.extensions.codehilite import CodeHiliteExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.tables import TableExtension
from markdown.extensions.toc import TocExtension


class DocsService:
    """Service for scanning and rendering documentation files."""
    
    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = Path(docs_dir)
        self.md = markdown.Markdown(
            extensions=[
                CodeHiliteExtension(css_class="highlight", guess_lang=True),
                FencedCodeExtension(),
                TableExtension(),
                TocExtension(permalink=True),
                'md_in_html',
            ],
            output_format='html5'
        )
    
    def get_tree(self) -> List[Dict[str, Any]]:
        """
        Recursively scan the docs directory for markdown files.
        Returns a nested tree structure suitable for frontend rendering.
        
        Returns:
            List of tree nodes, each with:
            - name: Display name (filename without .md)
            - path: Relative path from docs/ (for API calls)
            - type: 'file' or 'folder'
            - children: List of child nodes (for folders)
        """
        if not self.docs_dir.exists():
            return []
        
        return self._scan_directory(self.docs_dir)
    
    def _scan_directory(self, directory: Path) -> List[Dict[str, Any]]:
        """Recursively scan a directory and build tree nodes."""
        nodes = []
        
        try:
            entries = sorted(directory.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        except PermissionError:
            return nodes
        
        for entry in entries:
            # Skip hidden files/folders and __pycache__
            if entry.name.startswith('.') or entry.name.startswith('__'):
                continue
            
            if entry.is_dir():
                # Recursively scan subdirectory
                children = self._scan_directory(entry)
                # Only include folder if it has markdown files (directly or nested)
                if children:
                    nodes.append({
                        "name": entry.name,
                        "path": str(entry.relative_to(self.docs_dir)),
                        "type": "folder",
                        "children": children
                    })
            elif entry.suffix.lower() == '.md':
                # Add markdown file
                rel_path = entry.relative_to(self.docs_dir)
                nodes.append({
                    "name": entry.stem,  # Filename without extension
                    "path": str(rel_path),
                    "type": "file"
                })
        
        return nodes
    
    def render_doc(self, relative_path: str) -> Optional[Dict[str, Any]]:
        """
        Read a markdown file and convert it to HTML.
        
        Args:
            relative_path: Path relative to docs/ directory
            
        Returns:
            Dict with:
            - html: Rendered HTML content
            - title: Document title (from first H1 or filename)
            - toc: Table of contents HTML
            Or None if file doesn't exist or is outside docs/
        """
        # Security: Prevent path traversal
        try:
            full_path = (self.docs_dir / relative_path).resolve()
            # Ensure the resolved path is within docs directory
            if not str(full_path).startswith(str(self.docs_dir.resolve())):
                return None
        except (ValueError, OSError):
            return None
        
        if not full_path.exists() or not full_path.is_file():
            return None
        
        if full_path.suffix.lower() != '.md':
            return None
        
        try:
            content = full_path.read_text(encoding='utf-8')
        except (IOError, UnicodeDecodeError):
            return None
        
        # Reset the markdown instance for fresh conversion
        self.md.reset()
        
        # Convert markdown to HTML
        html = self.md.convert(content)
        
        # Extract title from first H1 or use filename
        title = self._extract_title(content, full_path.stem)
        
        # Get TOC if available
        toc = getattr(self.md, 'toc', '')
        
        return {
            "html": html,
            "title": title,
            "toc": toc,
            "path": relative_path
        }
    
    def _extract_title(self, content: str, fallback: str) -> str:
        """Extract title from markdown content (first H1) or use fallback."""
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        return fallback.replace('_', ' ').replace('-', ' ').title()


# Singleton instance
docs_service = DocsService()






