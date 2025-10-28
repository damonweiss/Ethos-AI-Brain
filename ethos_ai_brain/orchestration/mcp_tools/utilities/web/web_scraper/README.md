# Web Scraper UV Tool

Professional web scraping tool with CSS selectors and robust error handling.

## Features

- **CSS Selector Support**: Target specific elements with CSS selectors
- **Robust Error Handling**: Graceful handling of timeouts, connection errors, and HTTP errors
- **Custom Headers**: Support for custom HTTP headers
- **URL Validation**: Validate URLs without downloading full content
- **Modern Architecture**: UV-managed with proper isolation

## Tools

### scrape_webpage

Scrape content from web pages with optional CSS selectors.

**Parameters:**
- `url` (required): URL to scrape
- `selector` (optional): CSS selector for specific content
- `timeout` (optional): Request timeout in seconds (default: 10)
- `headers` (optional): Custom HTTP headers

**Examples:**

```python
# Scrape entire page
{"url": "https://example.com"}

# Scrape specific elements
{"url": "https://news.ycombinator.com", "selector": ".titleline > a"}

# With custom headers and timeout
{
    "url": "https://api.example.com/data",
    "timeout": 15,
    "headers": {"Authorization": "Bearer token123"}
}
```

### validate_url

Validate if a URL is accessible without downloading full content.

**Parameters:**
- `url` (required): URL to validate
- `timeout` (optional): Request timeout in seconds (default: 5)

## Installation

```bash
uv sync --project web_scraper_uv
```

## Usage

```bash
uv run --project web_scraper_uv python -c "from tool import scrape_webpage; print(scrape_webpage({'url': 'https://example.com'}))"
```
