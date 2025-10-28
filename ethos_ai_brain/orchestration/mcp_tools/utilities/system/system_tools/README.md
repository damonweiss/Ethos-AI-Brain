# System Tools

Basic system utilities for MCP server functionality.

## Features

- **Pure Python**: No external dependencies required
- **Lightweight**: Minimal resource usage
- **Reliable**: Basic system operations with robust error handling
- **UV-Managed**: Modern packaging with proper isolation

## Tools

### echo_message

Echo back a message for testing MCP communication and connectivity.

**Parameters:**
- `message` (optional): Message to echo back (default: "No message provided")

**Examples:**

```python
# Echo a message
{"message": "Hello MCP!"}

# Echo with default message
{}
```

### get_system_time

Get current system time and timestamp information.

**Parameters:**
- None required

**Returns:**
- ISO formatted timestamp
- Unix timestamp
- Server name
- Timezone information
- UTC timestamp

## Installation

```bash
uv sync --project utilities/system/system_tools
```

## Usage

```bash
uv run --project utilities/system/system_tools python -c "from tool import echo_message; print(echo_message({'message': 'Hello!'}))"
```
