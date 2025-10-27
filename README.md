# Ethos AI Brain

A Python library to create an effective AI brain for any project, maintained by Ethos Collaborative.

## Overview

Ethos AI Brain is a comprehensive framework for building intelligent systems with advanced reasoning capabilities. It provides a modular architecture for AI agents, knowledge management, meta-reasoning, and orchestration through various interfaces including Flask web services and MCP (Model Context Protocol) integration.

## Features

### Core Capabilities
- **AI Agent Framework**: Flexible agent architecture for various AI tasks
- **Meta-Reasoning Engine**: Advanced reasoning capabilities with adaptive analysis
- **Intent Analysis**: Sophisticated intent recognition and processing
- **Knowledge Management**: Structured knowledge storage and retrieval
- **Schema Management**: Dynamic schema handling and validation

### Orchestration & Integration
- **Flask Web Integration**: RESTful API endpoints for web applications
- **MCP Protocol Support**: Model Context Protocol client-server implementation
- **ZeroMQ Integration**: High-performance messaging through Ethos-ZeroMQ
- **Multi-Model Support**: Integration with OpenAI, Anthropic, Cohere, and other AI providers

### Visualization & Analysis
- **Network Visualization**: Graph-based knowledge representation
- **Cognitive Mapping**: Visual representation of reasoning processes
- **System Monitoring**: Built-in tools for system analysis and debugging

## Installation

### Prerequisites
- Python 3.13 or higher
- Git (for dependency installation)

### Install from Source
```bash
git clone https://github.com/damonweiss/Ethos-AI-Brain.git
cd Ethos-AI-Brain
pip install -e .
```

### Development Installation
For development with additional tools:
```bash
pip install -e .[dev]
```

## Quick Start

### Basic Usage
```python
from ethos_ai_brain.reasoning.meta_reasoning.reasoning_manager import ReasoningManager
from ethos_ai_brain.reasoning.intent_analysis.adaptive_intent_analyzer import AdaptiveIntentAnalyzer

# Initialize the reasoning manager
reasoning_manager = ReasoningManager()

# Initialize intent analyzer
intent_analyzer = AdaptiveIntentAnalyzer()

# Your AI brain is ready to use!
```

### Running the Demo
Explore the cognitive capabilities with the included demonstration:
```bash
python demo_cognitive_capabilities.py
```

### Starting Web Services
Launch the Flask web interface:
```bash
python start_flask.py
```

Launch the MCP server:
```bash
python start_mcp.py
```

## Project Structure

```
ethos_ai_brain/
├── config/           # Configuration management
├── core/             # Core AI agent and schema components
│   ├── ai_agent/     # AI agent framework
│   ├── prompt_manager/       # Prompt handling and routing
│   ├── schema_manager/       # Schema validation and management
│   └── schemas/      # Data schemas
├── knowledge/        # Knowledge management and visualization
├── orchestration/    # Service orchestration and integration
│   ├── mcp/          # Model Context Protocol implementation
│   └── flask_integration/    # Flask web service integration
└── reasoning/        # Advanced reasoning capabilities
    ├── inference_engines/    # Inference processing
    ├── intent_analysis/      # Intent recognition
    └── meta_reasoning/       # Meta-cognitive processes
```

## Configuration

The system uses environment variables for configuration. Copy `.env_sample` to `.env` and configure:

```bash
cp .env_sample .env
# Edit .env with your API keys and settings
```

## Testing

Run the test suite:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=ethos_ai_brain
```

## Development

### Code Quality
The project uses several tools for code quality:
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Format code:
```bash
black ethos_ai_brain/
isort ethos_ai_brain/
```

### Pre-commit Hooks
Install pre-commit hooks for automatic code quality checks:
```bash
pre-commit install
```

## Dependencies

### Core Dependencies
- **AI/ML**: OpenAI, Anthropic, Cohere, LiteLLM
- **Web Framework**: Flask, Flask-CORS
- **Data Processing**: Pydantic, PyYAML, NetworkX
- **Visualization**: Matplotlib
- **Messaging**: Ethos-ZeroMQ

### Development Dependencies
- **Testing**: pytest, pytest-cov
- **Documentation**: Sphinx, sphinx-rtd-theme
- **Code Quality**: black, isort, flake8, mypy
- **Development Tools**: Jupyter, IPython

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Damon Weiss**  
Email: damonweiss@ethoscollaborative.com  
Organization: Ethos Collaborative

## Repository

[https://github.com/damonweiss/Ethos-AI-Brain](https://github.com/damonweiss/Ethos-AI-Brain)

## Version

Current version: 0.0.2 (Alpha)

---

*This project is in active development. Features and APIs may change between versions.*
