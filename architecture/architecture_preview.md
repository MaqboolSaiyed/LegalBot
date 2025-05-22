# RAG Architecture Diagram

## System Overview

```mermaid
graph TB
    %% Main Components
    User[User Interface] --> Query[Query Input]
    Query --> QueryAgent[Query Agent]
    QueryAgent --> DocRetrieval[Document Retrieval]
    DocRetrieval --> ContextProc[Context Processing]
    ContextProc --> SumAgent[Summarization Agent]
    SumAgent --> Response[Response Generation]
    Response --> User

    %% Document Store
    subgraph DocumentStore[Document Store]
        LegalDocs[Legal Documents]
        ICAIDocs[ICAI Guidelines]
    end

    %% Query Agent Details
    subgraph QueryAgentDetails[Query Agent Components]
        QueryParse[Query Parsing]
        IntentRecog[Intent Recognition]
        QueryOpt[Query Optimization]
    end

    %% Document Retrieval Details
    subgraph RetrievalDetails[Retrieval Components]
        VectorDB[Vector Database]
        Embeddings[Document Embeddings]
        Similarity[Similarity Search]
    end

    %% Context Processing Details
    subgraph ContextDetails[Context Processing]
        WindowMgmt[Window Management]
        Relevance[Relevance Scoring]
        Aggregation[Information Aggregation]
    end

    %% Summarization Details
    subgraph SumDetails[Summarization Components]
        T5Model[T5 Model]
        PromptEng[Prompt Engineering]
        ResponseFormat[Response Formatting]
    end

    %% Connections
    QueryAgent --> QueryParse
    QueryParse --> IntentRecog
    IntentRecog --> QueryOpt
    QueryOpt --> DocRetrieval

    DocRetrieval --> VectorDB
    VectorDB --> Embeddings
    Embeddings --> Similarity
    Similarity --> ContextProc

    ContextProc --> WindowMgmt
    WindowMgmt --> Relevance
    Relevance --> Aggregation
    Aggregation --> SumAgent

    SumAgent --> T5Model
    T5Model --> PromptEng
    PromptEng --> ResponseFormat
    ResponseFormat --> Response

    %% Document Store Connections
    DocRetrieval --> LegalDocs
    DocRetrieval --> ICAIDocs

    %% Styling
    classDef primary fill:#1a365d,stroke:#2a4365,color:white
    classDef secondary fill:#2d3748,stroke:#4a5568,color:white
    classDef tertiary fill:#4a5568,stroke:#718096,color:white

    class User,Query,Response primary
    class QueryAgent,DocRetrieval,ContextProc,SumAgent secondary
    class QueryParse,IntentRecog,QueryOpt,VectorDB,Embeddings,Similarity,WindowMgmt,Relevance,Aggregation,T5Model,PromptEng,ResponseFormat tertiary
```

## How to Preview

1. Open this file in VS Code
2. Press `Ctrl+Shift+V` (Windows/Linux) or `Cmd+Shift+V` (Mac) to open the preview
3. The diagram will be rendered automatically

## Alternative Preview Methods

1. **Using Mermaid Preview Extension**:
   - Right-click on the file
   - Select "Open Preview to the Side"
   - The diagram will be rendered in real-time

2. **Using Command Palette**:
   - Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
   - Type "Markdown: Open Preview"
   - Select the option to open preview

## Export Options

Once you're happy with the diagram, you can:
1. Use the Mermaid Live Editor (https://mermaid.live) to export as PNG/SVG
2. Use the "Mermaid Preview" extension's export feature
3. Take a screenshot of the preview 