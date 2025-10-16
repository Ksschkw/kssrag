# Contributing to KSS RAG

## Welcome Contributors!

Thank you for your interest in contributing to KSS RAG! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Bug Reports](#bug-reports)
- [Feature Requests](#feature-requests)
- [Community](#community)

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to foster an open and welcoming environment.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- pip (Python package installer)

### First Time Contributors

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/kssrag.git
   cd kssrag
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/Ksschkw/kssrag.git
   ```

4. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/amazing-feature
   ```

## Development Setup

### Installation for Development

```bash
# Clone the repository
git clone https://github.com/Ksschkw/kssrag.git
cd kssrag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e .[dev,ocr,all]

# Install pre-commit hooks
pre-commit install
```

### Development Dependencies

The project uses several development tools:

```bash
# Code formatting
pip install black isort

# Linting
pip install flake8 pylint

# Type checking
pip install mypy

# Testing
pip install pytest pytest-cov pytest-mock

# Documentation
pip install mkdocs mkdocs-material
```

## Project Structure

```
kssrag/
├── core/                   # Core framework components
│   ├── chunkers.py         # Document segmentation
│   ├── vectorstores.py     # Vector database implementations
│   ├── retrievers.py       # Information retrieval
│   └── agents.py           # RAG orchestration
├── models/                 # LLM integrations
│   ├── openrouter.py       # OpenRouter API client
│   └── local_llms.py       # Local LLM implementations
├── utils/                  # Utility functions
│   ├── helpers.py          # Common utilities
│   ├── document_loaders.py # Document parsing
│   ├── ocr_loader.py       # OCR processing
│   └── preprocessors.py    # Text preprocessing
├── tests/                  # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── fixtures/           # Test fixtures
├── docs/                   # Documentation
├── examples/               # Usage examples
└── scripts/                # Development scripts
```

## Development Workflow

### 1. Sync with Upstream

```bash
# Fetch latest changes from upstream
git fetch upstream

# Merge changes into your branch
git merge upstream/main
```

### 2. Make Your Changes

Follow the coding standards and write tests for new functionality.

### 3. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=kssrag tests/

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

### 4. Code Quality Checks

```bash
# Format code
black kssrag/ tests/
isort kssrag/ tests/

# Lint code
flake8 kssrag/ tests/
pylint kssrag/

# Type checking
mypy kssrag/
```

### 5. Commit Your Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add new vector store implementation

- Implement ChromaVectorStore
- Add integration tests
- Update documentation

Closes #123"
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some specific conventions:

```python
# Imports: standard library, third-party, local
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from kssrag.core.vectorstores import BaseVectorStore

# Class names: CamelCase
class CustomVectorStore(BaseVectorStore):
    """Class docstring explaining purpose."""
    
    # Constants: UPPER_CASE
    DEFAULT_CHUNK_SIZE = 500
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize with configuration."""
        self.config = config or Config()
    
    # Method names: snake_case
    def add_documents(self, documents: List[Dict]) -> None:
        """Add documents to vector store.
        
        Args:
            documents: List of document dictionaries
            
        Raises:
            ValueError: If documents format is invalid
        """
        if not documents:
            raise ValueError("Documents cannot be empty")
        
        # Implementation here
```

### Type Hints

Use type hints for all function signatures and important variables:

```python
from typing import List, Dict, Optional, Union, Any

def process_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 500,
    metadata: Optional[Dict[str, str]] = None
) -> List[Dict[str, Union[str, Dict]]]:
    """Process documents with type hints."""
    # Implementation
```

### Docstring Format

Use Google-style docstrings:

```python
def retrieve_documents(query: str, top_k: int = 5) -> List[Document]:
    """Retrieve documents based on query.
    
    Args:
        query: Search query string
        top_k: Number of results to return (default: 5)
        
    Returns:
        List of relevant documents sorted by relevance
        
    Raises:
        VectorStoreError: If retrieval fails
        ValueError: If query is empty
        
    Example:
        >>> documents = retrieve_documents("machine learning", top_k=3)
        >>> len(documents)
        3
    """
```

## Testing

### Writing Tests

```python
# tests/unit/test_vectorstores.py
import pytest
from kssrag.core.vectorstores import BM25VectorStore

class TestBM25VectorStore:
    """Test suite for BM25VectorStore."""
    
    @pytest.fixture
    def sample_documents(self):
        return [
            {"content": "Machine learning is fascinating", "metadata": {"id": 1}},
            {"content": "Deep learning is a subset of ML", "metadata": {"id": 2}},
        ]
    
    def test_add_documents(self, sample_documents):
        """Test adding documents to store."""
        store = BM25VectorStore()
        store.add_documents(sample_documents)
        
        assert len(store.documents) == 2
        assert store.documents[0]["content"] == sample_documents[0]["content"]
    
    def test_retrieve_documents(self, sample_documents):
        """Test document retrieval."""
        store = BM25VectorStore()
        store.add_documents(sample_documents)
        
        results = store.retrieve("machine learning", top_k=1)
        
        assert len(results) == 1
        assert "machine learning" in results[0]["content"].lower()
```

### Test Guidelines

- Write tests for all new functionality
- Maintain test coverage above 80%
- Use descriptive test method names
- Include both unit and integration tests
- Mock external dependencies

## Documentation

### Updating Documentation

We use Markdown for documentation. When adding new features:

1. **Update README.md** if introducing major features
2. **Add API documentation** in `docs/api_reference.md`
3. **Create examples** in `examples/` directory
4. **Update docstrings** in code

### Building Documentation

```bash
# Build documentation locally
mkdocs serve

# Build for production
mkdocs build
```

## Pull Request Process

### 1. Create a Pull Request

1. Push your branch to your fork
2. Create a PR against the main repository's `main` branch
3. Fill out the PR template completely

### 2. PR Template

```markdown
## Description
Brief description of the changes

## Related Issue
Fixes # (issue number)

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Added unit tests
- [ ] Added integration tests
- [ ] Manual testing performed

## Documentation
- [ ] Updated README
- [ ] Updated API documentation
- [ ] Added code comments

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] Changes are backward compatible
```

### 3. Code Review

- Address review comments promptly
- Make requested changes
- Keep PR focused and manageable
- Squash commits if requested

### 4. Merge

- PRs require at least one approval
- All checks must pass
- Maintainer will merge the PR

## Bug Reports

### Reporting Bugs

Use the GitHub issue template and include:

```markdown
## Description
Clear description of the bug

## Steps to Reproduce
1. 
2. 
3. 

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: 
- Python Version:
- KSS RAG Version:
- Dependencies:

## Additional Context
Screenshots, logs, etc.
```

## Feature Requests

### Suggesting Features

```markdown
## Problem Statement
Clear description of the problem

## Proposed Solution
Description of proposed feature

## Alternatives Considered
Other solutions considered

## Additional Context
Use cases, examples, etc.
```

## Community

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and ideas
- **Documentation**: For usage guidance

### Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

---

Thank you for contributing to KSS RAG! Your efforts help make this project better for everyone.

---

These comprehensive documentation files provide enterprise-grade guidance for users, contributors, and maintainers. They maintain the professional tone and depth expected from a major technology company while being practical and actionable.