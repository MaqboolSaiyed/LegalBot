import os
import gradio as gr
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.tokenize import sent_tokenize
import warnings
import torch
import fitz  # PyMuPDF
import io

# Download required NLTK data
nltk.download('punkt', quiet=True)

class SimpleLegalAssistant:
    def __init__(self):
        """Initialize the Simple Legal Assistant with basic components."""
        print("🚀 Starting Simple Legal Assistant...")
        
        # Load the small sentence transformer model with CPU settings
        print("📚 Loading sentence transformer model...")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
        # Force model to use FP32
        self.model.to(torch.float32)
        
        # Initialize document storage
        self.documents = []
        self.chunks = []
        self.embeddings = []
        
        # Pre-load documents from the documents folder
        self.preload_documents()
        
        print("✅ Simple Legal Assistant initialized successfully!")

    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF using both PyMuPDF and PyPDF2 for better results"""
        try:
            # First try PyMuPDF (fitz) for better text extraction
            doc = fitz.open(file_path)
            text = ""
            
            for page in doc:
                # Get text with better formatting
                page_text = page.get_text("text")
                # Clean up the text
                page_text = re.sub(r'\s+', ' ', page_text)  # Replace multiple spaces with single space
                page_text = re.sub(r'([.!?])\s+', r'\1\n', page_text)  # Add newlines after sentences
                text += page_text + "\n"
            
            doc.close()
            
            # If PyMuPDF extraction is too short, try PyPDF2 as backup
            if len(text.strip()) < 100:
                reader = PdfReader(file_path)
                backup_text = ""
                for page in reader.pages:
                    backup_text += page.extract_text() + "\n"
                
                if len(backup_text) > len(text):
                    text = backup_text
            
            # Additional text cleaning
            text = re.sub(r'\n\s*\n', '\n\n', text)  # Remove multiple newlines
            text = re.sub(r'([A-Z])\s+([A-Z])', r'\1\2', text)  # Fix spacing in acronyms
            text = re.sub(r'(\d+)\s+(\d+)', r'\1\2', text)  # Fix spacing in numbers
            
            # Remove headers and footers
            text = re.sub(r'Page \d+ of \d+', '', text)
            text = re.sub(r'\d+\s*©.*?Guide to L itigation in India', '', text)
            text = re.sub(r'^\d+\s*', '', text, flags=re.MULTILINE)
            
            return text.strip()
            
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return ""

    def preload_documents(self):
        """Pre-load documents from the documents folder"""
        documents_dir = 'documents'
        if not os.path.exists(documents_dir):
            print(f"⚠️ Documents directory '{documents_dir}' not found. Creating it...")
            os.makedirs(documents_dir)
            return

        pdf_files = [f for f in os.listdir(documents_dir) if f.endswith('.pdf')]
        if not pdf_files:
            print("⚠️ No PDF files found in the documents folder.")
            return

        print(f"📚 Found {len(pdf_files)} PDF files. Loading documents...")
        
        for pdf_file in pdf_files:
            try:
                file_path = os.path.join(documents_dir, pdf_file)
                print(f"Processing file: {file_path}")
                
                # Extract text using improved method
                text = self.extract_text_from_pdf(file_path)
                
                if not text.strip():
                    print(f"Warning: No text extracted from {pdf_file}")
                    continue
                
                # Split into chunks using the chunk_text method
                chunks = self.chunk_text(text)
                
                if not chunks:
                    print(f"Warning: No chunks created from {pdf_file}")
                    continue
                
                # Create embeddings
                with torch.no_grad():
                    chunk_embeddings = self.model.encode(chunks, convert_to_tensor=True)
                    chunk_embeddings = chunk_embeddings.cpu().numpy()
                
                # Store as numpy arrays
                self.documents.append(text)
                self.chunks.extend(chunks)
                self.embeddings.extend(chunk_embeddings.tolist())
                
                print(f"✅ Successfully processed {pdf_file}")
                
            except Exception as e:
                print(f"❌ Error processing {pdf_file}: {str(e)}")
                continue
        
        if self.documents:
            print(f"✅ Successfully loaded {len(self.documents)} documents")
        else:
            print("⚠️ No documents were successfully loaded")

    def chunk_text(self, text):
        """Split text into chunks of approximately 500 characters"""
        # Split into sentences first
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= 500:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def expand_query(self, query):
        """Expand the query with relevant legal terms and synonyms"""
        # Common legal terms and their related concepts
        legal_terms = {
            'file': ['filing', 'submit', 'lodge', 'register', 'initiate'],
            'case': ['lawsuit', 'suit', 'petition', 'proceeding', 'matter'],
            'court': ['judiciary', 'tribunal', 'bench', 'judge'],
            'procedure': ['process', 'steps', 'method', 'protocol'],
            'jurisdiction': ['authority', 'power', 'competence'],
            'india': ['indian', 'national', 'domestic'],
            'legal': ['lawful', 'legitimate', 'statutory'],
            'document': ['paper', 'record', 'file', 'petition'],
            'evidence': ['proof', 'testimony', 'documentation'],
            'trial': ['hearing', 'proceeding', 'case'],
            'appeal': ['review', 'challenge', 'petition'],
            'judgment': ['decision', 'order', 'ruling'],
            'plaintiff': ['complainant', 'petitioner', 'applicant'],
            'defendant': ['respondent', 'accused', 'opponent']
        }
        
        # Expand query with related terms
        expanded_terms = [query.lower()]
        words = query.lower().split()
        
        for word in words:
            if word in legal_terms:
                expanded_terms.extend(legal_terms[word])
        
        return ' '.join(expanded_terms)

    def search_documents(self, query, top_k=5):
        """Search for relevant document chunks"""
        if not self.embeddings:
            return "No documents loaded. Please add PDF files to the 'documents' folder."
        
        try:
            # Expand the query
            expanded_query = self.expand_query(query)
            
            # Create query embedding
            with torch.no_grad():
                query_embedding = self.model.encode([expanded_query], convert_to_tensor=True)
                query_embedding = query_embedding.cpu().numpy()
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top k results with higher similarity threshold
            similarity_threshold = 0.25  # Lowered threshold for better recall
            top_indices = np.argsort(similarities)[::-1]
            filtered_indices = [idx for idx in top_indices if similarities[idx] > similarity_threshold][:top_k]
            
            if not filtered_indices:
                return "No relevant information found for your query."
            
            # Group similar chunks
            results = []
            used_chunks = set()
            
            for idx in filtered_indices:
                chunk = self.chunks[idx]
                # Skip if chunk is too similar to already selected chunks
                if any(self.compute_similarity(chunk, used_chunk) > 0.8 for used_chunk in used_chunks):
                    continue
                    
                results.append({
                    'chunk': chunk,
                    'similarity': float(similarities[idx])
                })
                used_chunks.add(chunk)
            
            return results
        except Exception as e:
            print(f"Error in search: {str(e)}")
            return []

    def compute_similarity(self, text1, text2):
        """Compute similarity between two text chunks"""
        try:
            with torch.no_grad():
                embedding1 = self.model.encode([text1], convert_to_tensor=True)
                embedding2 = self.model.encode([text2], convert_to_tensor=True)
                similarity = cosine_similarity(embedding1.cpu().numpy(), embedding2.cpu().numpy())[0][0]
                return float(similarity)
        except:
            return 0.0

    def generate_summary(self, chunks):
        """Generate a summary from the retrieved chunks"""
        if not chunks:
            return "No relevant information found."
        
        try:
            # Combine chunks
            text = " ".join([chunk['chunk'] for chunk in chunks])
            
            # Split into sentences
            sentences = sent_tokenize(text)
            
            # Score sentences based on multiple factors
            sentence_scores = []
            for sentence in sentences:
                score = 0
                
                # Length score (prefer medium-length sentences)
                length = len(sentence)
                if 50 <= length <= 200:
                    score += 0.3
                elif length > 200:
                    score += 0.1
                
                # Position score (prefer sentences at the beginning)
                position = sentences.index(sentence)
                if position < len(sentences) // 3:
                    score += 0.2
                
                # Keyword score (prefer sentences with important legal terms)
                legal_terms = ['court', 'jurisdiction', 'file', 'case', 'procedure', 'law', 'legal', 
                             'petition', 'plaintiff', 'defendant', 'judge', 'trial', 'evidence',
                             'india', 'indian', 'filing', 'steps', 'process', 'document']
                term_count = sum(1 for term in legal_terms if term.lower() in sentence.lower())
                score += term_count * 0.15  # Increased weight for legal terms
                
                # Remove sentences with low information content
                if len(sentence.split()) < 5 or score < 0.2:
                    continue
                    
                sentence_scores.append((sentence, score))
            
            # Get top sentences
            top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:4]
            
            # Combine into summary
            summary = " ".join([s[0] for s in top_sentences])
            
            return summary
        except Exception as e:
            print(f"Error in summary generation: {str(e)}")
            return "Error generating summary. Please try again."

    def process_query(self, query):
        """Process a user query"""
        try:
            # Check if documents are loaded
            if not self.documents:
                return "No documents are loaded. Please add PDF files to the 'documents' folder."
            
            # Check if query is provided
            if not query or not query.strip():
                return "Please enter a query."
            
            # Search for relevant chunks
            search_results = self.search_documents(query)
            if isinstance(search_results, str):  # Error message returned
                return search_results
            
            # Generate summary
            summary = self.generate_summary(search_results)
            
            # Format response with better structure
            response = "Based on the available legal documents:\n\n"
            response += f"Summary:\n{summary}\n\n"
            
            if search_results:
                response += "Key Points:\n"
                for i, result in enumerate(search_results, 1):
                    # Clean up the text by removing extra spaces and special characters
                    cleaned_chunk = ' '.join(result['chunk'].split())
                    # Remove page numbers and headers
                    cleaned_chunk = re.sub(r'\d+\s+©.*?Guide to L itigation in India', '', cleaned_chunk)
                    cleaned_chunk = re.sub(r'^\d+\s+', '', cleaned_chunk)
                    response += f"{i}. {cleaned_chunk}\n"
            
            return response
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}"

# Create a wrapper function for Gradio
def process_query_wrapper(query):
    try:
        # Create new assistant instance for each query
        assistant = SimpleLegalAssistant()
        
        # Process the query
        result = assistant.process_query(query)
        
        # Check if result is an error message
        if isinstance(result, str) and ("Error" in result or "No documents" in result):
            return result
            
        return result
    except Exception as e:
        print(f"Error in wrapper: {str(e)}")
        return f"An error occurred: {str(e)}"

# Create the interface
iface = gr.Interface(
    fn=process_query_wrapper,
    inputs=[
        gr.Textbox(label="Enter your query", placeholder="Ask a question about the documents...")
    ],
    outputs=gr.Textbox(label="Response"),
    title="Simple Legal Assistant",
    description="Ask questions about the pre-loaded documents in the 'documents' folder.",
    examples=None
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()
