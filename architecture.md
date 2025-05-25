# Simple Legal Assistant - Architecture

## System Overview

```mermaid
graph TB
    %% Main Components
    User[User Interface] --> Query[User Query]
    Query --> Pipeline[Multi-Agent Pipeline]
    Pipeline --> QueryAgent[Query Agent]
    Pipeline --> SumAgent[Summarization Agent]
    QueryAgent --> Response[Final Response]
    SumAgent --> Response
    Response --> User

    %% Query Agent Details
    subgraph QueryAgentDetails[Query Agent Components]
        Model[all-MiniLM-L6-v2 Model]
        DocLoad[Document Loading]
        Chunking[Text Chunking]
        Search[Cosine Similarity Search]
    end

    %% Summarization Agent Details
    subgraph SumAgentDetails[Summarization Components]
        Extract[Key Point Extraction]
        Scoring[Sentence Scoring]
        Summary[Summary Generation]
    end

    %% Document Store
    subgraph DocumentStore[Document Store]
        PDFs[PDF Documents]
        Chunks[Text Chunks]
        Embeddings[Document Embeddings]
    end

    %% Connections
    QueryAgent --> Model
    Model --> DocLoad
    DocLoad --> PDFs
    PDFs --> Chunking
    Chunking --> Chunks
    Chunks --> Embeddings
    Embeddings --> Search

    SumAgent --> Extract
    Extract --> Scoring
    Scoring --> Summary

    %% Styling
    classDef primary fill:#1a365d,stroke:#2a4365,color:white
    classDef secondary fill:#2d3748,stroke:#4a5568,color:white
    classDef tertiary fill:#4a5568,stroke:#718096,color:white

    class User,Query,Response primary
    class Pipeline,QueryAgent,SumAgent secondary
    class Model,DocLoad,Chunking,Search,Extract,Scoring,Summary,PDFs,Chunks,Embeddings tertiary
```

## Workflow Description

### 1. User Interaction
- User submits a query through the Gradio interface
- Query is passed to the Multi-Agent Pipeline

### 2. Query Agent Processing
1. **Document Loading**
   - Loads PDF documents (max 50 pages each)
   - Processes documents in batches of 10
   - Shows progress with emoji indicators

2. **Text Chunking**
   - Splits text into 500-character chunks
   - Uses sentence boundaries
   - Maintains readability

3. **Search Process**
   - Uses all-MiniLM-L6-v2 for embeddings
   - Performs cosine similarity search
   - Returns top 3 relevant chunks

### 3. Summarization Agent Processing
1. **Key Point Extraction**
   - Analyzes retrieved chunks
   - Identifies important sentences
   - Considers sentence length and position

2. **Sentence Scoring**
   - Scores based on keyword matches
   - Considers sentence length
   - Combines multiple scoring factors

3. **Summary Generation**
   - Selects top 4 sentences
   - Maintains coherence
   - Formats response

### 4. Response Generation
- Combines search results and summary
- Formats for user readability
- Returns through Gradio interface

## Memory Optimization

```mermaid
graph LR
    %% Memory Optimization Components
    CPU[CPU Processing] --> Batch[Batch Processing]
    Batch --> Chunk[Efficient Chunking]
    Chunk --> Embed[Lightweight Embeddings]
    Embed --> Search[Simple Search]
    Search --> Summary[Basic Summarization]

    %% Optimization Details
    subgraph Optimizations[Memory Optimizations]
        BatchSize[10 Documents/Batch]
        PageLimit[50 Pages/PDF]
        ChunkSize[500 Chars/Chunk]
        TopK[Top 3 Results]
    end

    %% Styling
    classDef primary fill:#1a365d,stroke:#2a4365,color:white
    classDef secondary fill:#2d3748,stroke:#4a5568,color:white

    class CPU,Batch,Chunk,Embed,Search,Summary primary
    class BatchSize,PageLimit,ChunkSize,TopK secondary
```

## Error Handling

```mermaid
graph TB
    %% Error Handling Flow
    Input[User Input] --> Validation[Input Validation]
    Validation --> Processing[Document Processing]
    Processing --> Search[Document Search]
    Search --> Summary[Summary Generation]
    Summary --> Output[Final Output]

    %% Error Recovery
    Validation -- Error --> ErrorMsg[Error Message]
    Processing -- Error --> Retry[Retry Processing]
    Search -- Error --> Fallback[Fallback Response]
    Summary -- Error --> Default[Default Summary]

    %% Styling
    classDef primary fill:#1a365d,stroke:#2a4365,color:white
    classDef error fill:#c53030,stroke:#9b2c2c,color:white

    class Input,Validation,Processing,Search,Summary,Output primary
    class ErrorMsg,Retry,Fallback,Default error
```

## Performance Metrics

- **Memory Usage**: ~4GB RAM
- **Processing Time**: < 2 minutes
- **Response Time**: < 5 seconds
- **Document Limit**: 50 pages per PDF
- **Batch Size**: 10 documents
- **Chunk Size**: 500 characters
- **Top Results**: 3 chunks
- **Summary Length**: 4 sentences 