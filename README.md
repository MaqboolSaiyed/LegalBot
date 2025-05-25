---
title: Simple Indian Legal & Accounting Assistant
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.7.1
app_file: app.py
pinned: false
---

# Simple Indian Legal & Accounting Assistant

A lightweight AI assistant that provides information about Indian litigation processes and ICAI guidelines using a simplified RAG (Retrieval-Augmented Generation) implementation.

## Features

- Memory-efficient document retrieval
- Simple, clear explanations of legal concepts
- Support for Indian litigation procedures and ICAI guidelines
- Fast response generation
- User-friendly interface
- CPU-only operation

## Technical Architecture

The system uses a simplified RAG architecture that combines:
1. Document Retrieval: Efficient semantic search using cosine similarity
2. Text Processing: Simple sentence-based chunking
3. Response Generation: Extractive summarization

## Project Structure

```
.
├── app.py                    # Main application file
├── documents/               # PDF documents
│   ├── litigation_guide.pdf
│   └── icai_guidelines.pdf
├── requirements.txt         # Dependencies
└── README.md               # Documentation
```

## Implementation Details

### 1. Document Processing
- Simple sentence-based chunking (500 characters per chunk)
- 50-page limit per PDF
- Batch processing (10 documents at a time)

### 2. Search and Retrieval
- Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- Cosine similarity for document matching
- Top-k retrieval (k=3)

### 3. Summarization
- Extractive summarization based on keyword matching
- Top 4 sentences selection
- Simple scoring mechanism

### 4. Memory Optimization
- CPU-only processing
- Batch processing
- Efficient document chunking
- Progress indicators

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your PDF documents in the `documents` folder:
   - `litigation_guide.pdf`
   - `icai_guidelines.pdf`

3. Run the application:
```bash
python app.py
```

## Example Queries

- "What are the steps to file a lawsuit in India?"
- "What documents are required for filing a case?"
- "What are the ICAI guidelines for chartered accountants?"
- "Explain the general litigation process in India"

## Dependencies

- sentence-transformers==2.2.2
- gradio==4.7.1
- PyMuPDF==1.23.5
- scikit-learn==1.3.0
- nltk==3.8.1
- numpy==1.24.3
- torch==2.0.1+cpu
- transformers==4.33.2

## Contributors

- Maqbool Saiyed - Lead Developer

## License

This project is open source and available under the MIT License.