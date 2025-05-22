import logging
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from utils.document_loader import get_document_text, DOCUMENTS_DIR, initialize_documents
import os
import pickle
import time # Import the time module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryAgent:
    def __init__(self):
        """Initialize the QueryAgent with a vector store for document retrieval."""
        logger.info("Initializing Query Agent...")

        # Initialize the sentence transformer model
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

        # Initialize or load the FAISS index and documents (which now include headers)
        self.index = None
        self.documents = [] # This will store tuples of (header, chunk_content)

        # Always initialize knowledge base to ensure correct format
        self.initialize_knowledge_base()

        logger.info("Query Agent initialized successfully.")

    def initialize_knowledge_base(self):
        """Initialize the knowledge base with documents, ensuring correct format."""
        try:
            # Ensure documents are initialized (creates sample files if they don't exist)
            initialize_documents()

            # Always create a new knowledge base to ensure correct format
            logger.info("Creating new knowledge base...")

            # Load documents and split into tagged chunks
            self.documents = []
            for doc_name in ["litigation_guide", "icai_guidelines"]:
                content = get_document_text(doc_name)
                if content:
                    # Split content into tagged chunks using the updated method
                    # The _split_into_chunks method now returns List[tuple[str, str]]
                    tagged_chunks = self._split_into_chunks(content)
                    self.documents.extend(tagged_chunks)

            if not self.documents:
                raise Exception("No documents found in knowledge base.")

            # Create embeddings from chunk content (the second element of the tuple)
            logger.info("Creating embeddings for document chunks...")
            # Extract only the chunk content for embedding
            chunk_contents = [chunk for header, chunk in self.documents]
            embeddings = self.model.encode(chunk_contents)

            # Create FAISS index
            logger.info("Creating FAISS index...")
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))

            # Save the index and the documents (including headers)
            index_dir = os.path.join(DOCUMENTS_DIR, "faiss_index")
            os.makedirs(index_dir, exist_ok=True)
            faiss.write_index(self.index, os.path.join(index_dir, "index.faiss"))
            # Save the documents list (containing tuples)
            with open(os.path.join(index_dir, "documents.pkl"), 'wb') as f:
                pickle.dump(self.documents, f)

            logger.info(f"Knowledge base initialized with {len(self.documents)} tagged chunks.")

        except Exception as e:
            logger.error(f"Error initializing knowledge base: {e}")
            self.index = None
            self.documents = []
            # Re-raise the exception to indicate initialization failure
            raise

    def _split_into_chunks(self, text: str, chunk_size: int = 500) -> List[tuple[str, str]]:
        """Split text into overlapping chunks and tag them with their section header."""
        sections_with_content = []
        current_section_content_lines = []
        current_header = "General Information" # Default header
        import re
        markdown_header_pattern = re.compile(r"^##\s+(.+)$") # Regex for '## Header Text'

        lines = text.split('\n')

        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue

            match = markdown_header_pattern.match(stripped_line)
            if match:
                # If there's content for the previous section, save it
                if current_section_content_lines:
                    sections_with_content.append((current_header, '\n'.join(current_section_content_lines)))
                # Start a new section
                current_header = match.group(1).strip() # Use the captured header text
                current_section_content_lines = [stripped_line] # Include the '## Header' line itself
            else:
                # Add line to the current section's content
                current_section_content_lines.append(line) # Append original line to preserve formatting/indentation

        # Add the last section's content
        if current_section_content_lines:
            sections_with_content.append((current_header, '\n'.join(current_section_content_lines)))

        # Process each section into chunks with header tag
        chunks = []
        for header, content in sections_with_content:
            if not content.strip():
                continue

            # Split content into words
            words = content.split()

            # Create chunks with overlap
            for i in range(0, len(words), chunk_size // 2): # Use chunk_size // 2 for 50% overlap
                chunk_words = words[i:i + chunk_size]
                if chunk_words:
                    chunk_content = ' '.join(chunk_words)
                    # Append the header as the first element of the tuple
                    chunks.append((header, chunk_content))

        return chunks

    def process(self, query: str) -> Optional[str]:
        """Process a query using semantic search and return relevant information from the most relevant section.

        Returns the combined content of relevant chunks from the dominant section.
        """
        if not self.index or not self.documents:
            logger.warning("Knowledge base (vector store or documents) is not available. Cannot process query.")
            return None

        try:
            # Simulate retrieval time
            time.sleep(1) # Add a 1-second delay
            logger.info("Simulating knowledge base retrieval...")

            # Encode the query
            query_embedding = self.model.encode([query])[0]

            # Search the index - retrieve several relevant chunks
            k = 5 # Retrieve top 5 chunks
            distances, indices = self.index.search(query_embedding.reshape(1, -1).astype('float32'), k)

            # Get the relevant chunks (which are tuples of (header, chunk_content))
            relevant_chunks_with_headers = [self.documents[i] for i in indices[0]]

            # Analyze headers of retrieved chunks to find the most represented section
            header_counts = {}
            for header, chunk_content in relevant_chunks_with_headers:
                header_counts[header] = header_counts.get(header, 0) + 1

            # Determine the dominant header (most frequently occurring in the top chunks)
            dominant_header = None
            if header_counts:
                dominant_header = max(header_counts, key=header_counts.get)

            # Collect content only from chunks belonging to the dominant header
            combined_content = ""
            if dominant_header:
                content_from_dominant_section = [chunk for header, chunk in relevant_chunks_with_headers if header == dominant_header]
                combined_content = "\n\n".join(content_from_dominant_section).strip()
                logger.info(f"Identified dominant section: '{dominant_header}'. Combining {len(content_from_dominant_section)} relevant chunks.")
            else:
                # Fallback if no dominant header is found (e.g., chunks are spread across multiple sections)
                logger.warning("No dominant section header found in retrieved chunks. Combining all retrieved chunks.")
                combined_content = "\n\n".join([chunk for header, chunk in relevant_chunks_with_headers]).strip()
                # Add a generic header for the summarization agent
                if combined_content:
                     combined_content = "General Information:\n\n" + combined_content

            if not combined_content:
                 logger.warning("Combined content is empty after processing.")
                 return None # Return None if no content could be combined

            return combined_content

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return None

if __name__ == '__main__':
    # This is for testing the QueryAgent directly
    print("Testing QueryAgent...")
    agent = QueryAgent()

    if agent.index:
        test_queries = [
            "What are the steps to file a lawsuit in India?",
            "What documents are required for filing a case?",
            "What are the key ICAI guidelines for chartered accountants?",
            "how to file a petition",
            "ICAI ethics"
        ]

        for query in test_queries:
            # Use a simple print for debugging f-string issue
            print(f"\nSending test query: {query}")
            response = agent.process(query)
            if response:
                # Print only first 500 chars to keep output clean
                print(f"Response received. Starting with: {response[:500]}...")
            else:
                print("No response generated.")
    else:
        print("QueryAgent could not be initialized properly (vector store missing).")
