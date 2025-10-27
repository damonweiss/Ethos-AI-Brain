# MCP Tools Documentation

Total tools: 6

## Relationship_Inference Tools

### infer_relationships

**Description:** Intelligently infer relationships between entities using meta-reasoning heuristics

**Parameters:**

- `entities` (array): List of entities to analyze for relationships
- `relationship_types` (array): Optional focus on specific relationship types
- `confidence_threshold` (number): Minimum confidence for including relationships
- `heuristic_strategies` (array): Heuristic strategies to apply

---

### infer_relationships

**Description:** Intelligently discover meaningful relationships between entities using heuristic analysis

**Parameters:**

- `entities` (array): List of entities to analyze for relationships
- `relationship_types` (array): Optional focus on specific relationship types
- `confidence_threshold` (number): Minimum confidence for including relationships

---

## Utilities Tools

### echo_message

**Description:** Echo back a message for testing MCP communication and connectivity

**Tags:** testing, communication, echo, debug

**Author:** Ethos Collaborative

**Version:** 1.0.0

**Parameters:**

- `message` (string): Message to echo back

**Examples:**

*Echo a simple message*

Input: `{'message': 'Hello MCP!'}`

Output: `{'echo': 'Hello MCP!', 'timestamp': '2025-01-28T12:00:00.000Z', 'processed_by': 'ethos-mcp-server'}`

*Echo with no message (uses default)*

Input: `{}`

Output: `{'timestamp': '2025-01-28T12:00:00.000Z', 'processed_by': 'ethos-mcp-server'}`

---

### get_system_time

**Description:** Get current system time and timestamp information

**Tags:** time, system, timestamp, utility

**Author:** Ethos Collaborative

**Version:** 1.0.0

**Examples:**

*Get current system time*

Input: `{}`

Output: `{'timestamp': '2025-01-28T12:00:00.000Z', 'unix_timestamp': 1737288000.0, 'server': 'ethos-mcp-server', 'timezone': 'UTC'}`

---

## Web_Tools Tools

### scrape_webpage

**Description:** Scrape content from web pages with optional CSS selectors and robust error handling

**Tags:** web, scraping, content, html, css-selectors

**Author:** Ethos Collaborative

**Version:** 2.0.0

**Parameters:**

- `url` (string): URL to scrape
- `selector` (string): CSS selector for specific content
- `timeout` (integer): Request timeout in seconds
- `headers` (object): Custom HTTP headers

**Examples:**

*Scrape entire page content*

Input: `{'url': 'https://example.com'}`

Output: `{'url': 'https://example.com', 'content': 'Example Domain...', 'status_code': 200, 'content_type': 'text/html'}`

*Scrape specific elements with CSS selector*

Input: `{'url': 'https://news.ycombinator.com', 'selector': '.titleline > a'}`

Output: `{'url': 'https://news.ycombinator.com', 'content': ['Article 1', 'Article 2', 'Article 3'], 'selector': '.titleline > a', 'elements_found': 3}`

*Scrape with custom timeout and headers*

Input: `{'url': 'https://api.example.com/data', 'timeout': 15, 'headers': {'Authorization': 'Bearer token123'}}`

Output: `{'url': 'https://api.example.com/data', 'content': 'API response data...', 'status_code': 200}`

---

### validate_url

**Description:** Validate if a URL is accessible without downloading full content

**Tags:** web, validation, url, connectivity

**Author:** Ethos Collaborative

**Version:** 1.0.0

**Parameters:**

- `url` (string): URL to validate
- `timeout` (integer): Request timeout in seconds

**Examples:**

*Validate a URL*

Input: `{'url': 'https://example.com'}`

Output: `{'url': 'https://example.com', 'valid': True, 'status_code': 200, 'content_type': 'text/html'}`

---

