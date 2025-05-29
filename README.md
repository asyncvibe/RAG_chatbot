# RAG Chatbot

A chatbot implementation using Retrieval-Augmented Generation (RAG) that can understand and respond to queries about documents.

## Description

This project implements a chatbot that uses RAG (Retrieval-Augmented Generation) to provide accurate responses based on the content of provided documents. The chatbot can process PDF documents and engage in conversations about their content.

## Features

- PDF document processing
- Natural language understanding
- Context-aware responses
- RAG-based information retrieval
- Conversational interface

## Getting Started

### Prerequisites

- Python 3.8 or higher
- UV package manager

### Installation

1. Clone the repository:

   ```bash
   git clone [repository-url]
   cd rag-chatbot
   ```

2. Install dependencies using UV:
   ```bash
   uv pip install -r requirements.txt
   ```

### Usage

Run the chatbot:

```bash
python main.py
```

## Project Structure

```
rag-chatbot/
├── Eat_That_Frog.pdf    # Sample document for chatbot
├── main.py              # Main application file
├── pyproject.toml       # Project dependencies and configuration
└── README.md           # Project documentation
```

## TODO List

The following features are planned for future implementation:

1. Authentication

   - User login/signup system
   - Session management
   - Access control

2. Document Upload

   - Web interface for document upload
   - Document validation
   - Format conversion support

3. Multiple Document Upload

   - Batch upload functionality
   - Document organization
   - Document metadata management

4. Unit Tests

   - Test coverage for core functionality
   - Integration tests
   - Performance testing

5. Internet Access for Chatbot
   - Web search capabilities
   - Real-time information retrieval
   - Fact verification

## Contributing

Contributions are welcome! Please feel free to submit pull requests.
