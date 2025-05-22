---
title: Indian Legal & Accounting Assistant
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.30.0
app_file: app.py
pinned: false
---

# Indian Legal & Accounting Assistant

A specialized AI assistant that provides information about Indian litigation processes and ICAI guidelines using advanced RAG (Retrieval-Augmented Generation) technology.

## Features

- Accurate retrieval of legal information from trusted sources
- Clear, simplified explanations of complex legal concepts
- Support for Indian litigation procedures and ICAI guidelines
- Fast and efficient response generation
- User-friendly interface

## Technical Architecture

The system uses a RAG (Retrieval-Augmented Generation) architecture, which combines:
1. Document Retrieval: Efficiently searches through legal documents
2. Context Processing: Analyzes and extracts relevant information
3. Response Generation: Creates clear, concise answers

For detailed architecture documentation, please refer to:
- `architecture/RAG_Architecture.md`: Complete system architecture
- `architecture/Development_Challenges.md`: Challenges faced and solutions implemented

## Project Structure

```
.
├── app.py                    # Main application file
├── agents/                   # AI agents
│   ├── __init__.py
│   ├── query_agent.py       # Query processing agent
│   └── summarization_agent.py # Text summarization agent
├── architecture/            # Architecture documentation
│   ├── RAG_Architecture.md  # System architecture details
│   └── Development_Challenges.md # Development challenges
├── utils/                  # Utility functions
└── requirements.txt        # Dependencies
```

## Contributors

- Maqbool Saiyed - Lead Developer

## License

This project is open source and available under the MIT License.