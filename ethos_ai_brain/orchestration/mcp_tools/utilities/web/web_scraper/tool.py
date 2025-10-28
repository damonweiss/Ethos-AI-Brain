"""
Professional Web Scraper Tool

UV-managed web scraping tool with CSS selectors, error handling, and modern architecture.
"""

from typing import Dict
import logging

logger = logging.getLogger(__name__)


def _import_dependencies():
    """Import external dependencies with graceful fallback."""
    try:
        import requests
        from bs4 import BeautifulSoup
        return requests, BeautifulSoup
    except ImportError as e:
        logger.error(f"Missing dependencies for web scraper: {e}")
        return None, None


def scrape_webpage(params: Dict) -> Dict:
    """
    Scrape content from a webpage with CSS selector support.
    
    Args:
        params: Dictionary containing:
            - url (str): URL to scrape
            - selector (str, optional): CSS selector for specific content
            - timeout (int, optional): Request timeout in seconds (default: 10)
            - headers (dict, optional): Custom HTTP headers
            
    Returns:
        Dict containing scraped content or error message
    """
    requests, BeautifulSoup = _import_dependencies()
    
    if not requests or not BeautifulSoup:
        return {
            "error": "Missing required dependencies",
            "install_command": "uv sync --project utilities/web/web_scraper",
            "dependencies": ["requests>=2.31.0", "beautifulsoup4>=4.12.0", "lxml>=4.9.0"]
        }
    
    # Validate required parameters
    url = params.get('url')
    if not url:
        return {
            "error": "URL parameter is required",
            "example": {"url": "https://example.com"}
        }
    
    # Extract optional parameters
    selector = params.get('selector')
    timeout = params.get('timeout', 10)
    custom_headers = params.get('headers', {})
    
    # Default headers for better compatibility
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        **custom_headers
    }
    
    try:
        # Make HTTP request
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        
        # Parse HTML content with fallback parser
        try:
            soup = BeautifulSoup(response.content, 'lxml')
        except:
            # Fallback to built-in parser if lxml not available
            soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract content based on selector
        if selector:
            elements = soup.select(selector)
            if elements:
                content = [elem.get_text().strip() for elem in elements]
                # Remove empty strings
                content = [text for text in content if text]
            else:
                content = []
        else:
            # Extract all text content
            content = soup.get_text().strip()
        
        # Prepare response
        result = {
            "url": url,
            "content": content,
            "status_code": response.status_code,
            "content_type": response.headers.get('content-type', 'unknown'),
            "content_length": len(response.content),
            "encoding": response.encoding or 'unknown'
        }
        
        # Add selector info if used
        if selector:
            result.update({
                "selector": selector,
                "elements_found": len(content) if isinstance(content, list) else 1
            })
        
        return result
        
    except requests.exceptions.Timeout:
        return {
            "error": f"Request timeout after {timeout} seconds",
            "url": url,
            "suggestion": "Try increasing the timeout parameter"
        }
    except requests.exceptions.ConnectionError:
        return {
            "error": f"Failed to connect to {url}",
            "url": url,
            "suggestion": "Check if the URL is accessible and your internet connection"
        }
    except requests.exceptions.HTTPError as e:
        return {
            "error": f"HTTP error: {e}",
            "url": url,
            "status_code": response.status_code if 'response' in locals() else None
        }
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "url": url,
            "type": type(e).__name__
        }


def validate_url(params: Dict) -> Dict:
    """
    Validate if a URL is accessible without scraping content.
    
    Args:
        params: Dictionary containing:
            - url (str): URL to validate
            - timeout (int, optional): Request timeout in seconds
            
    Returns:
        Dict containing validation result
    """
    requests, _ = _import_dependencies()
    
    if not requests:
        return {
            "error": "Missing requests dependency",
            "install_command": "uv sync --project utilities/web/web_scraper"
        }
    
    url = params.get('url')
    if not url:
        return {"error": "URL parameter is required"}
    
    timeout = params.get('timeout', 5)
    
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        
        return {
            "url": url,
            "valid": True,
            "status_code": response.status_code,
            "content_type": response.headers.get('content-type', 'unknown'),
            "server": response.headers.get('server', 'unknown'),
            "final_url": response.url if response.url != url else None
        }
        
    except Exception as e:
        return {
            "url": url,
            "valid": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


# Tool metadata for dynamic discovery
TOOLS = [
    {
        "name": "scrape_webpage",
        "namespace": "domain.web",
        "category": "web_scraping",
        "description": "Scrape content from web pages with optional CSS selectors and robust error handling",
        "function": scrape_webpage,
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "format": "uri",
                    "description": "URL to scrape",
                    "examples": ["https://example.com", "https://news.ycombinator.com"]
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector for specific content",
                    "examples": [".title", "#main-content", "h1, h2, h3", ".titleline > a"]
                },
                "timeout": {
                    "type": "integer",
                    "description": "Request timeout in seconds",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 60
                },
                "headers": {
                    "type": "object",
                    "description": "Custom HTTP headers",
                    "examples": [{"Authorization": "Bearer token"}]
                }
            },
            "required": ["url"],
            "additionalProperties": False
        },
        "returns": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "content": {"type": ["string", "array"]},
                "status_code": {"type": "integer"},
                "content_type": {"type": "string"},
                "content_length": {"type": "integer"},
                "encoding": {"type": "string"},
                "selector": {"type": "string"},
                "elements_found": {"type": "integer"},
                "error": {"type": "string"}
            }
        },
        "tags": ["web", "scraping", "content", "html", "css-selectors"],
        "author": "Ethos Collaborative",
        "version": "2.0.0",
        "category": "web_tools",
        "examples": [
            {
                "description": "Scrape entire page content",
                "input": {"url": "https://example.com"},
                "output": {
                    "url": "https://example.com",
                    "content": "Example Domain...",
                    "status_code": 200,
                    "content_type": "text/html"
                }
            },
            {
                "description": "Scrape specific elements with CSS selector",
                "input": {
                    "url": "https://news.ycombinator.com", 
                    "selector": ".titleline > a"
                },
                "output": {
                    "url": "https://news.ycombinator.com",
                    "content": ["Article 1", "Article 2", "Article 3"],
                    "selector": ".titleline > a",
                    "elements_found": 3
                }
            },
            {
                "description": "Scrape with custom timeout and headers",
                "input": {
                    "url": "https://api.example.com/data",
                    "timeout": 15,
                    "headers": {"Authorization": "Bearer token123"}
                },
                "output": {
                    "url": "https://api.example.com/data",
                    "content": "API response data...",
                    "status_code": 200
                }
            }
        ]
    },
    {
        "name": "validate_url",
        "namespace": "domain.web",
        "category": "web_validation",
        "description": "Validate if a URL is accessible without downloading full content",
        "function": validate_url,
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "format": "uri",
                    "description": "URL to validate"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Request timeout in seconds",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 30
                }
            },
            "required": ["url"],
            "additionalProperties": False
        },
        "returns": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "valid": {"type": "boolean"},
                "status_code": {"type": "integer"},
                "content_type": {"type": "string"},
                "server": {"type": "string"},
                "final_url": {"type": "string"},
                "error": {"type": "string"}
            }
        },
        "tags": ["web", "validation", "url", "connectivity"],
        "author": "Ethos Collaborative",
        "version": "1.0.0",
        "category": "web_tools",
        "examples": [
            {
                "description": "Validate a URL",
                "input": {"url": "https://example.com"},
                "output": {
                    "url": "https://example.com",
                    "valid": True,
                    "status_code": 200,
                    "content_type": "text/html"
                }
            }
        ]
    }
]
